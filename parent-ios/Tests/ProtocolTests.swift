import XCTest
import CryptoKit
@testable import PuppyParent

final class ProtocolTests: XCTestCase {
    let now = Date(timeIntervalSince1970: 1_900_000_000)
    var key: Curve25519.Signing.PrivateKey { try! .init(rawRepresentation: Data(0..<32)) }
    var pairing: Pairing { Pairing(serverID: "home-mac", deviceID: "puppy-1", serverPublicKey: key.publicKey.rawRepresentation.base64EncodedString()) }
    func command(_ kind: CommandKind = .status) throws -> Command { try .make(kind: kind, pairing: pairing, sourceID: "iphone-test", now: now) }
    func receipt(_ command: Command, state: ReceiptState = .succeeded, updated: Double = 1_900_000_001) -> Receipt {
        Receipt(version: 1, requestID: command.id, serverID: command.serverID, sourceID: command.sourceID, deviceID: command.deviceID,
                kind: command.kind.rawValue, state: state, updatedAt: updated,
                result: ReceiptResult(ok: true, status: nil, media: nil, capturedAt: nil, error: nil), error: nil)
    }

    func testSigningUsesExactPayloadAndMatchesEd25519KnownKey() throws {
        let command = try command()
        let signed = try Wire.sign(command, key: key)
        let decoded: Command = try Wire.verified(signed, publicKey: pairing.serverPublicKey, as: Command.self)
        XCTAssertEqual(decoded, command)
        let raw = try JSONDecoder().decode(Envelope.self, from: signed)
        XCTAssertTrue(key.publicKey.isValidSignature(Data(base64Encoded: raw.signature)!, for: Data(base64Encoded: raw.payload)!))
        XCTAssertEqual(key.publicKey.rawRepresentation.base64EncodedString(), "A6EHv/POEL4dcN0Y50vAmWfk1jCbpQ1fHdyGZBJVMbg=")
    }

    func testTamperedPayloadAndWrongKeyRejected() throws {
        let signed = try Wire.sign(command(), key: key)
        let envelope = try JSONDecoder().decode(Envelope.self, from: signed)
        let tampered = try Wire.encode(Envelope(payload: Data("{}".utf8).base64EncodedString(), signature: envelope.signature))
        XCTAssertThrowsError(try Wire.verified(tampered, publicKey: pairing.serverPublicKey, as: Command.self))
        XCTAssertThrowsError(try Wire.verified(signed, publicKey: Curve25519.Signing.PrivateKey().publicKey.rawRepresentation.base64EncodedString(), as: Command.self))
    }

    func testActualPythonProtocolFixtureVerifiesInCryptoKit() throws {
        let file = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "python-receipt", withExtension: "json"))
        let value = try Wire.verified(Data(contentsOf: file), publicKey: pairing.serverPublicKey, as: Receipt.self)
        XCTAssertEqual(value.requestID, "39ae28fe-c934-4fb6-8803-1e5b0b22706c")
        XCTAssertEqual(value.result?.status?.petName, "豆豆")
        XCTAssertEqual(value.result?.status?.volume, 0)
        XCTAssertEqual(value.state, .succeeded)
    }

    func testCommandSchemaWhitelistAndTTL() throws {
        let value = try Command.make(kind: .action, action: "wave", pairing: pairing, sourceID: "iphone-test", now: now)
        let object = try JSONSerialization.jsonObject(with: Wire.encode(value)) as! [String: Any]
        XCTAssertEqual(Set(object.keys), ["version", "request_id", "source_id", "server_id", "device_id", "kind", "params", "created_at", "expires_at"])
        XCTAssertEqual(value.expiresAt - value.createdAt, 300)
        XCTAssertEqual(value.requestID, value.requestID.lowercased())
        XCTAssertThrowsError(try Command.make(kind: .action, action: "laser", pairing: pairing, sourceID: "iphone-test", now: now))
        XCTAssertThrowsError(try Command.make(kind: .status, action: "wave", pairing: pairing, sourceID: "iphone-test", now: now))
        XCTAssertThrowsError(try Command.make(kind: .status, pairing: pairing, sourceID: "iphone-test", now: now, ttl: 301))
    }

    func testOptionalPhotoDescriptionRemainsBoundToSinglePhoto() throws {
        let data = Data(#"{"ok":true,"description":{"photo_id":"photo-1","reply":" 看见一张桌子。 ","uncertain":true,"scope":"single_photo"}}"#.utf8)
        let value = try JSONDecoder().decode(ReceiptResult.self, from: data)
        let signed = try Wire.sign(value, key: key)
        let verified = try Wire.verified(signed, publicKey: pairing.serverPublicKey, as: ReceiptResult.self)
        XCTAssertEqual(verified.description?.displayReply, "看见一张桌子。")
        XCTAssertEqual(verified.description?.uncertain, true)
        XCTAssertEqual(verified.description?.photoID, "photo-1")
        let absent = try JSONDecoder().decode(ReceiptResult.self, from: Data(#"{"ok":true}"#.utf8))
        XCTAssertNil(absent.description)
        let unavailable = try JSONDecoder().decode(ReceiptResult.self, from: Data(#"{"ok":true,"description":{"ok":false,"error":"description_unavailable"}}"#.utf8))
        XCTAssertEqual(unavailable.ok, true)
        XCTAssertNil(unavailable.description?.displayReply)
        XCTAssertEqual(unavailable.description?.error, "description_unavailable")
        XCTAssertNil(PhotoDescription(photoID: nil, reply: "整个家都安全", uncertain: false, scope: "whole_home", ok: true, error: nil).displayReply)
    }

    func testExecutedReceiptMustBindFullTarget() throws {
        let request = try command()
        let value = receipt(request)
        XCTAssertNoThrow(try value.validate(command: request, now: now.addingTimeInterval(2)))
        XCTAssertThrowsError(try value.validate(command: command(), now: now.addingTimeInterval(2)))
        let other = Receipt(version: 1, requestID: request.id, serverID: request.serverID, sourceID: "another-phone", deviceID: request.deviceID,
                            kind: "status", state: .succeeded, updatedAt: 1_900_000_001, result: value.result, error: nil)
        XCTAssertThrowsError(try other.validate(command: request, now: now.addingTimeInterval(2)))
    }

    func testSignedRejectionCompatibilityAndTerminalStateNeverRegresses() throws {
        let request = try command()
        let rejection = Receipt(version: 1, requestID: request.id, serverID: request.serverID, sourceID: nil, deviceID: nil,
                                kind: nil, state: .rejected, updatedAt: 1_900_000_001, result: nil, error: "expired")
        XCTAssertNoThrow(try rejection.validate(command: request, now: now.addingTimeInterval(2)))
        let completed = receipt(request)
        XCTAssertFalse(try receipt(request, state: .accepted, updated: 1_900_000_009).supersedes(completed))
        XCTAssertThrowsError(try rejection.supersedes(completed))
        XCTAssertFalse(try receipt(request, state: .accepted).supersedes(receipt(request, state: .executing)))
    }

    func testStaleOrUnknownObservationNeverShowsOnline() {
        let stale = DeviceStatus(online: true, observedAt: now.timeIntervalSince1970 - 61, volume: 0, battery: nil, petName: nil)
        let snapshot = Snapshot(version: 1, serverID: "home-mac", deviceID: "puppy-1", publishedAt: now.timeIntervalSince1970, status: stale)
        XCTAssertFalse(snapshot.online(at: now))
        XCTAssertFalse(DeviceStatus(online: true, observedAt: nil, volume: 0, battery: nil, petName: nil).fresh(at: now))
        let current = DeviceStatus(online: true, observedAt: now.timeIntervalSince1970, volume: 0, battery: nil, petName: nil)
        XCTAssertTrue(Snapshot(version: 1, serverID: "home-mac", deviceID: "puppy-1", publishedAt: now.timeIntervalSince1970, status: current).online(at: now))
    }

    func testMediaHashLengthAndPathTraversalRejected() throws {
        let data = Data("jpeg fixture bytes".utf8)
        let id = "39ae28fe-c934-4fb6-8803-1e5b0b22706c"
        let media = MediaReference(file: "media/\(id).jpg", sha256: Wire.hash(data), bytes: data.count)
        XCTAssertNoThrow(try media.validate(data, requestID: id))
        XCTAssertThrowsError(try media.validate(data + Data([0]), requestID: id))
        XCTAssertThrowsError(try MediaReference(file: "../private/key", sha256: Wire.hash(data), bytes: data.count).validate(data, requestID: id))
        XCTAssertThrowsError(try MediaReference(file: media.file, sha256: String(repeating: "0", count: 64), bytes: data.count).validate(data, requestID: id))
    }

    func testBatteryTelemetryDecodesWithoutClaimingChargeDetection() throws {
        let current = try JSONDecoder().decode(Battery.self, from: Data(
            #"{"level":100,"charging_known":false,"servo_voltage_v":8.4}"#.utf8))
        XCTAssertEqual(current.level, 100)
        XCTAssertNil(current.charging)
        XCTAssertEqual(current.servoVoltageV, 8.4)
        XCTAssertEqual(current.chargingKnown, false)
        let legacy = try JSONDecoder().decode(Battery.self, from: Data(
            #"{"level":100,"charging":false}"#.utf8))
        XCTAssertNil(legacy.servoVoltageV)
        XCTAssertNil(legacy.chargingKnown)
    }

    func testSubmissionExpiryIsNotClaimedAsServerCompletion() throws {
        let request = try command()
        let record = RequestRecord(command: request, publicationConfirmed: true, receipt: nil)
        XCTAssertEqual(record.title(now: now), "已提交")
        XCTAssertEqual(record.title(now: now.addingTimeInterval(301)), "已过期 · 等待回执")
        for state in [ReceiptState.accepted, .executing] {
            let waiting = RequestRecord(command: request, publicationConfirmed: true, receipt: receipt(request, state: state))
            XCTAssertEqual(waiting.title(now: now.addingTimeInterval(301)), "已过期 · 结果待核实")
        }
        let done = RequestRecord(command: request, publicationConfirmed: true, receipt: receipt(request))
        XCTAssertEqual(done.title(now: now.addingTimeInterval(301)), "已完成")
    }

    func testDirectoryProducerConsumerAndCorruptPhotoIsolation() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathComponent("PuppyCompanion")
        defer { try? FileManager.default.removeItem(at: root.deletingLastPathComponent()) }
        for directory in ["receipts", "status", "media"] {
            try FileManager.default.createDirectory(at: root.appendingPathComponent(directory), withIntermediateDirectories: true)
        }
        let request = try Command.make(kind: .photo, pairing: pairing, sourceID: "iphone-test")
        let signed = try Wire.sign(request, key: key)
        try DirectoryIO.publish(signed, root: root, command: request)
        XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("commands/\(request.id).json")), signed)
        XCTAssertThrowsError(try DirectoryIO.publish(signed, root: root, command: request))
        let data = Data("test media".utf8)
        let media = MediaReference(file: "media/\(request.id).jpg", sha256: Wire.hash(data), bytes: data.count)
        let done = Receipt(version: 1, requestID: request.id, serverID: request.serverID, sourceID: request.sourceID,
                           deviceID: request.deviceID, kind: "photo", state: .succeeded, updatedAt: Date().timeIntervalSince1970,
                           result: ReceiptResult(ok: true, status: nil, media: media, capturedAt: Date().timeIntervalSince1970, error: nil), error: nil)
        try Wire.sign(done, key: key).write(to: root.appendingPathComponent("receipts/\(request.id).json"))
        try data.write(to: root.appendingPathComponent(media.file))
        let record = RequestRecord(command: request, publicationConfirmed: true, receipt: nil)
        let update = try DirectoryReader.read(root: root, pairing: pairing, records: [record])
        XCTAssertEqual(update.receipts[request.id]?.state, .succeeded)
        XCTAssertEqual(update.photos[request.id], data)
        try (data + Data([0])).write(to: root.appendingPathComponent(media.file))
        let corrupt = try DirectoryReader.read(root: root, pairing: pairing, records: [record])
        XCTAssertNil(corrupt.photos[request.id])
        XCTAssertEqual(corrupt.receipts[request.id]?.state, .succeeded)
        XCTAssertTrue(corrupt.notes.contains("照片校验失败，未显示"))
    }

    func testDirectoryTimeoutKeepsOneWorkerAndDoesNotQueueMore() async throws {
        let io = DirectoryIO()
        let started = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let first = Task {
            try await io.run(timeout: 2) { _ in
                started.signal()
                release.wait()
                return 1
            }
        }
        XCTAssertEqual(started.wait(timeout: .now() + 1), .success)
        do {
            _ = try await first.value
            XCTFail("hanging operation must time out")
        } catch { XCTAssertEqual(error.localizedDescription, ClientError.timeout.localizedDescription) }
        do {
            _ = try await io.run { _ in 2 }
            XCTFail("must not queue another potentially hung worker")
        } catch { XCTAssertEqual(error.localizedDescription, ClientError.busy.localizedDescription) }
        release.signal()
    }
}
