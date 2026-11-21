# Local Puppy networking changes

This is a local override of registry component `78/esp-ml307` version 3.6.5.
ESP-IDF gives a component in the project's `components/` directory precedence
over the downloaded registry copy. The rest of this directory is the upstream
Apache-2.0 release (see `LICENSE`).

The Puppy firmware runs a wake-word model, camera and animation engine before
Wi-Fi connects. On the tested ESP32-S3, the largest free internal RAM block
after connection was about 1 KB, so the upstream `EspTcp::Connect()` could not
create its 4 KB receive task. Upstream ignored that creation failure and
reported a successful TCP connection; the WebSocket handshake then timed out.
The local `EspTcp` uses a PSRAM task stack on ESP32-S3 when enabled, checks task
creation and releases that stack with `vTaskDeleteWithCaps()`.

The upstream WebSocket sent the handshake before registering the TCP receive
callback. A fast LAN server could respond before the callback existed. The
local version registers callbacks before starting TCP and sending the request.

TCP sockets use `TCP_NODELAY` and a five-second send timeout. A failed or zero
length send terminates the connection with `shutdown()`; it must not leave a
partial WebSocket frame for the next command to append to. WebSocket writes
only succeed when the transport accepts the entire frame.

`EspTcp::Disconnect()` now joins a receive task even after a peer has cleared
the connected flag. Socket shutdown wakes blocked I/O; the descriptor is only
closed after receive callbacks and any sender finish. If a callback takes
more than ten seconds, shutdown logs a warning and keeps waiting rather than
freeing an object still in use. Receive callbacks may request a disconnect,
but connection setup and object destruction belong to the owning task.
Destroying `EspTcp` from its own callback is explicitly unsupported.

The WebSocket destructor always disconnects and destroys its transport before
destroying callback targets and receive buffers. Its connection flag and the
shared TCP connection flag are atomic. Resetting a WebSocket connection also
clears its old handshake and fragmentation state; the send mutex covers the
fragmentation flag as well as the transport write.

These changes affect users of the shared WebSocket component, including
non-Puppy boards. The receive-task join and socket shutdown implementation is
specific to `EspTcp`; SSL and modem transports keep their own disconnect
implementations. Only the ESP32-S3 Puppy configuration has been build checked
for these local changes; SSL/modem runtime behavior has not been validated.

ESP-IDF requires flash and NVS operations to run on a task with an internal
stack. Puppy control messages schedule those operations on the main task. Do
not add flash access to callbacks executed by the PSRAM-backed TCP task.

## Companion firmware safety checks

The matching Puppy board changes keep motor-stall handling in the independent
motor control task. A detected stall latches a RAM mask, turns off the laser,
stops new actions and repeatedly disables torque on all five motors. Display
updates may wait for the application task; unloading does not. Neither a new
action nor the physical action button releases this latch. There is no automatic
restart or new maintenance API; a hardware restart currently clears the RAM
latch and requires the parent to clear the physical obstruction first.

The authenticated control channel caches the most recent 64 completed action
and stop receipts in PSRAM. Identical IDs with identical canonical content
return the saved receipt without repeating motion. Reusing an ID with different
content is rejected. These receipts survive WebSocket reconnects but not reboot
or eviction. They do not replace server-side durable IDs and expiry checks.

The local board registers only Wave, Sit, Stretch, Hug and stop as MCP motor
tools. Legacy laser, walking, raw angles, calibration, BLE and action-loop tools
are not registered. Local MCP volume is limited to integer 0–60 and cannot
raise a muted volume above zero; only the authenticated parent control channel
can unmute. User-only MCP maintenance registration is limited to read-only
system information, since the upstream user-only flag does not authenticate
tool calls. Reboot, firmware replacement and external upload/download tools
are not exposed through this local MCP path.
The separate audio protocol's `system/reboot` command is also rejected in local
companion mode so it cannot bypass the motor latch.

Host checks compiled the actual production safety function bodies with motor
and socket stubs, using AddressSanitizer and UndefinedBehaviorSanitizer. They
covered duplicate/conflicting motion IDs, reordered JSON fields, invalid command
fields, sustained torque-off and action rejection after a stall, and MCP mute
and volume limits. These checks and the ESP32-S3 firmware build pass; physical
torque-off timing, camera/network faults and reboot behavior still require
device acceptance. No hardware was flashed during these checks.

The local Puppy board now bounds each permitted Wave/Sit/Stretch/Hug sequence
with a 15-second monotonic deadline in the independent motor task. The deadline
reports `failed` and resets the sequence; only `ACTION_DONE` can report
`completed`. Duplicate request IDs do not extend the deadline. This is a
conservative safety ceiling, not a measured servo completion time; the real
sequence timing and physical stop latency still need device acceptance.

Control-channel and legacy safe MCP actions are tagged as gateway-owned. The
motor task reads the control socket's atomic connected flag and cancels those
actions on a detected disconnect; the socket disconnect callback does the same
cleanup. The control task clears its protected socket pointer before destruction,
and transport teardown joins old callbacks before reconnecting. Duplicate cleanup
cannot replace an existing terminal result or cancel a subsequent local button
action. An undetected network black hole is still bounded by the action deadline.
Offline button actions have the same deadline but survive gateway disconnects.

Wi-Fi provisioning start/end callbacks no longer start Keep_Sit or Sit_Reset.
Neither connection timeout nor successful reconfiguration initiates motion.
No motor angle, calibration parameter, tilt threshold, audio behavior or saved
volume changed. Ten host regression groups now include actual allowed sequence
completion, exact deadline failure, reconnect replay, concurrent disconnect/local
button dispatch, provisioning callbacks, and the existing stall/mute checks.
