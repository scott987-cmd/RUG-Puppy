#!/usr/bin/env python3
"""Compile the real EspSsl and HttpClient against deterministic host I/O stubs.

No ESP board, Internet, camera or audio is used. AF_UNIX socket pairs exercise
descriptor ownership; fake TLS returns partial writes/WANT/error. These tests
do not establish certificate validation, on-device RAM or handshake reliability.
In this IDF version an async connect step can select for 5 seconds while holding
the I/O lock; DNS also follows lwIP's own resolver timeouts. Only established
read/write retries are nonblocking with cancellable waits outside the I/O lock.
Requires Python 3 and a C++17 host compiler (CXX overrides clang++).
PUPPY_HOST_SANITIZERS=thread runs the same races with ThreadSanitizer instead of
the default address/undefined sanitizers.
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
COMPONENT = ROOT / "components/78__esp-ml307"

RUNTIME = r'''
#pragma once
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <cerrno>
#include <cstdio>
#include <sys/socket.h>
#include <fcntl.h>
#include <unistd.h>
using namespace std::chrono_literals;
using BaseType_t = int;
using EventBits_t = unsigned;
using TickType_t = unsigned;
constexpr int pdFALSE=0, pdTRUE=1, pdPASS=1, ESP_OK=0;
constexpr int MALLOC_CAP_SPIRAM=1, MALLOC_CAP_8BIT=2;
#define pdMS_TO_TICKS(ms) (ms)
#define configASSERT(x) assert(x)
#define ESP_LOGI(...) ((void)0)
#define ESP_LOGD(...) ((void)0)
#define ESP_LOGE(...) ((void)0)
#define ESP_LOGW(...) ((void)0)
struct Event { std::mutex m; std::condition_variable cv; unsigned bits=0; };
using EventGroupHandle_t = Event*;
inline std::atomic<int> deleted_events{0}, wait_timeouts{0}, waits{0};
inline bool create_event_ok=true;
inline Event* xEventGroupCreate(){return create_event_ok?new Event:nullptr;}
inline void vEventGroupDelete(Event* e){++deleted_events; delete e;}
inline unsigned xEventGroupSetBits(Event* e,unsigned b){std::lock_guard<std::mutex> l(e->m);e->bits|=b;e->cv.notify_all();return e->bits;}
inline unsigned xEventGroupClearBits(Event* e,unsigned b){std::lock_guard<std::mutex> l(e->m);auto old=e->bits;e->bits&=~b;return old;}
inline unsigned xEventGroupGetBits(Event* e){std::lock_guard<std::mutex> l(e->m);return e->bits;}
inline unsigned xEventGroupWaitBits(Event* e,unsigned b,int clear,int all,unsigned ms){
    ++waits;
    std::unique_lock<std::mutex> l(e->m);
    auto ready=[&]{return all ? (e->bits&b)==b : bool(e->bits&b);};
    // Accelerate only the 10-second join warning; a callback remains blocked
    // until the test releases it, even across multiple simulated timeouts.
    if(!e->cv.wait_for(l,std::chrono::milliseconds(ms==10000?20:ms),ready))++wait_timeouts;
    auto result=e->bits;if(clear)e->bits&=~b;return result;
}
struct Task { std::thread worker; };
using TaskHandle_t = Task*;
inline thread_local Task* current_task=reinterpret_cast<Task*>(1);
inline std::vector<std::unique_ptr<Task>> tasks;
inline bool create_ok=true;
inline int stack_caps=0;
inline Task* xTaskGetCurrentTaskHandle(){return current_task;}
inline int xTaskCreate(void(*f)(void*),const char*,unsigned,void* arg,int,Task** out){
    if(!create_ok)return 0;
    auto task=std::make_unique<Task>();auto ptr=task.get();*out=ptr;
    task->worker=std::thread([=]{current_task=ptr;f(arg);});tasks.push_back(std::move(task));return pdPASS;
}
inline int xTaskCreateWithCaps(void(*f)(void*),const char* n,unsigned s,void* a,int p,Task** o,int caps){
    stack_caps=caps;return xTaskCreate(f,n,s,a,p,o);
}
inline void vTaskDelete(void*){}
inline void vTaskDeleteWithCaps(void*){}
inline void join_tasks(){for(auto& t:tasks)if(t->worker.joinable())t->worker.join();tasks.clear();}
inline std::atomic<int64_t> extra_us{0};
inline int64_t esp_timer_get_time(){return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()+extra_us;}
inline void until(std::function<bool()> f){auto end=std::chrono::steady_clock::now()+2s;while(!f()){assert(std::chrono::steady_clock::now()<end);std::this_thread::sleep_for(1ms);}}
struct Gate {
    std::mutex m;std::condition_variable cv;bool entered=false,released=false;
    void block(){std::unique_lock<std::mutex> l(m);entered=true;cv.notify_all();cv.wait(l,[&]{return released;});}
    void wait(){std::unique_lock<std::mutex> l(m);assert(cv.wait_for(l,2s,[&]{return entered;}));}
    void release(){std::lock_guard<std::mutex> l(m);released=true;cv.notify_all();}
};
'''

TLS = r'''
#pragma once
#include "runtime.h"
constexpr int ESP_TLS_ERR_SSL_WANT_READ=-101, ESP_TLS_ERR_SSL_WANT_WRITE=-102;
using esp_err_t=int;
using esp_tls_error_handle_t=int;
struct esp_tls_cfg_t { int(*crt_bundle_attach)(void*)=nullptr;bool non_block=false;int timeout_ms=0; };
struct esp_tls_t {int fd=-1,peer=-1;};
inline int esp_crt_bundle_attach(void*){return 0;}
inline std::atomic<int> destroyed_tls{0}, write_calls{0}, connect_calls{0};
inline std::function<int(esp_tls_t*,const char*,size_t)> write_hook;
inline std::atomic<bool> handshake_stalls{false}, handshake_fails{false};
inline std::mutex received_mutex;
inline std::deque<std::string> received;
inline std::string sent_wire;
inline int last_fd=-1;
inline esp_tls_t* esp_tls_init(){auto* t=new esp_tls_t;int pair[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,pair)==0);t->fd=pair[0];t->peer=pair[1];last_fd=t->fd;return t;}
inline int esp_tls_conn_new_async(const char*,int,int,const esp_tls_cfg_t* c,esp_tls_t*){
    ++connect_calls;assert(c->non_block&&c->timeout_ms>0&&c->crt_bundle_attach);return handshake_fails?-1:handshake_stalls?0:1;
}
inline int esp_tls_get_conn_sockfd(esp_tls_t* t,int* fd){*fd=t->fd;return 0;}
inline int esp_tls_get_error_handle(esp_tls_t*,int* h){*h=1;return 0;}
inline int esp_tls_get_and_clear_last_error(int,int*,int*){return 71;}
inline int esp_tls_conn_destroy(esp_tls_t* t){++destroyed_tls;assert(close(t->fd)==0);assert(close(t->peer)==0);delete t;return 0;}
inline int esp_tls_conn_read(esp_tls_t*,void* p,size_t n){
    std::lock_guard<std::mutex> l(received_mutex);
    if(received.empty())return ESP_TLS_ERR_SSL_WANT_READ;
    auto s=std::move(received.front());received.pop_front();assert(s.size()<=n);memcpy(p,s.data(),s.size());return s.size();
}
inline int esp_tls_conn_write(esp_tls_t* t,const void* p,size_t n){
    ++write_calls;int result=write_hook?write_hook(t,static_cast<const char*>(p),n):int(n);
    if(result>0)sent_wire.append(static_cast<const char*>(p),result);return result;
}
inline void deliver(std::string s){std::lock_guard<std::mutex> l(received_mutex);received.push_back(std::move(s));}
inline void reset_tls(){join_tasks();assert(received.empty());write_hook={};sent_wire.clear();write_calls=0;connect_calls=0;extra_us=0;handshake_stalls=false;handshake_fails=false;create_ok=true;}
'''

SSL_TEST = r'''
#include "esp_ssl.h"
#include "runtime.h"
int main(){
  { create_event_ok=false;EspSsl ssl;assert(!ssl.Connect("test",443));assert(ssl.GetLastError()==ENOMEM);create_event_ok=true;
  } puts("PASS SSL: event allocation failure rejects connect without dereferencing null");
  { reset_tls();create_ok=false;int before=destroyed_tls;EspSsl ssl;
    assert(!ssl.Connect("test",443));assert(!ssl.connected());assert(ssl.GetLastError()==ENOMEM);assert(destroyed_tls==before+1);
    create_ok=true;assert(ssl.Connect("test",443));ssl.Disconnect();assert(destroyed_tls==before+2);
  } puts("PASS SSL: receive-task creation failure rolls back and permits reconnect");
  { reset_tls();EspSsl ssl;Gate callback;std::atomic<bool> done=false;int before=destroyed_tls;
    ssl.OnStream([&](auto&){callback.block();});assert(ssl.Connect("test",443));int fd=last_fd;deliver("x");callback.wait();
    int timeouts=wait_timeouts;std::thread owner([&]{ssl.Disconnect();done=true;});until([&]{return wait_timeouts>=timeouts+2;});
    assert(!done&&destroyed_tls==before);assert(fcntl(fd,F_GETFD)!=-1);
    int probe=open("/dev/null",O_RDONLY);assert(probe>=0&&probe!=fd);
    callback.release();owner.join();assert(done&&destroyed_tls==before+1);assert(fcntl(probe,F_GETFD)!=-1);close(probe);
    ssl.Disconnect();assert(destroyed_tls==before+1);
  } puts("PASS SSL: timeout never frees callback state or closes/reuses its fd early");
  { reset_tls();EspSsl ssl;std::atomic<bool> callback_done=false;std::atomic<int> disconnected=0;
    ssl.OnStream([&](auto&){assert(ssl.Send("reply")==5);ssl.Disconnect();callback_done=true;});
    ssl.OnDisconnected([&]{++disconnected;});assert(ssl.Connect("test",443));deliver("x");until([&]{return callback_done.load();});
    ssl.Disconnect();assert(disconnected==1&&sent_wire=="reply");
  } puts("PASS SSL: receive callback can send and request disconnect without self-join");
  { reset_tls();EspSsl ssl;assert(ssl.Connect("test",443));write_hook=[](auto*,auto*,size_t){std::this_thread::sleep_for(1ms);return 1;};
    std::thread a([&]{assert(ssl.Send("AAAA")==4);});std::thread b([&]{assert(ssl.Send("BBBB")==4);});a.join();b.join();
    assert(sent_wire=="AAAABBBB"||sent_wire=="BBBBAAAA");ssl.Disconnect();
  } puts("PASS SSL: multiple writers serialize complete partially-written frames");
  { reset_tls();EspSsl ssl;assert(ssl.Connect("test",443));int attempts=0;
    write_hook=[&](auto*,auto*,size_t){return ++attempts==1?2:-77;};assert(ssl.Send("partial")==-1);assert(!ssl.connected());
    int calls=write_calls;assert(ssl.Send("next")==-1&&write_calls==calls);assert(sent_wire=="pa");ssl.Disconnect();
  } puts("PASS SSL: partial failure invalidates stream and rejects subsequent frames");
  { reset_tls();EspSsl ssl;assert(ssl.Connect("test",443));std::atomic<int> result{0};
    write_hook=[](auto*,auto*,size_t){return ESP_TLS_ERR_SSL_WANT_WRITE;};
    std::thread sender([&]{result=ssl.Send("blocked");});until([&]{return write_calls>=2;});
    auto start=std::chrono::steady_clock::now();ssl.Disconnect();sender.join();assert(result==-1);assert(std::chrono::steady_clock::now()-start<1s);
  } puts("PASS SSL: disconnect cancels WANT_WRITE outside the I/O lock and joins sender");
  { reset_tls();EspSsl ssl;assert(ssl.Connect("test",443));
    write_hook=[](auto*,auto*,size_t){extra_us+=6000000;return ESP_TLS_ERR_SSL_WANT_READ;};
    assert(ssl.Send("timeout")==-1);assert(!ssl.connected()&&ssl.GetLastError()==ETIMEDOUT);ssl.Disconnect();
  } puts("PASS SSL: total send deadline covers WANT_READ and invalidates stream");
  { reset_tls();EspSsl ssl;handshake_stalls=true;int before=destroyed_tls;
    auto connecting=std::async(std::launch::async,[&]{return ssl.Connect("test",443);});until([&]{return connect_calls>=2;});
    extra_us+=6000000;assert(!connecting.get());assert(ssl.GetLastError()==ETIMEDOUT&&destroyed_tls==before+1);
    handshake_stalls=false;assert(ssl.Connect("test",443));ssl.Disconnect();
  } puts("PASS SSL: stalled handshake has overall deadline and cleans up for retry");
  { reset_tls();EspSsl ssl;handshake_stalls=true;int before=destroyed_tls;
    auto connecting=std::async(std::launch::async,[&]{return ssl.Connect("test",443);});until([&]{return connect_calls>=2;});
    ssl.Disconnect();assert(!connecting.get()&&destroyed_tls==before+1);
  } puts("PASS SSL: owner cancellation drains an in-progress handshake once");
  join_tasks();
#if CONFIG_IDF_TARGET_ESP32S3 && CONFIG_SPIRAM_ALLOW_STACK_EXTERNAL_MEMORY
  assert(stack_caps==(MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT));
#endif
}
'''

NETWORK = r'''
#pragma once
#include "tcp.h"
#include <memory>
class NetworkInterface { public: virtual ~NetworkInterface()=default;
  virtual std::unique_ptr<Tcp> CreateTcp(int)=0;
  virtual std::unique_ptr<Tcp> CreateSsl(int)=0;
};
'''

HTTP_TEST = r'''
#include "runtime.h"
#include "http_client.h"
#include "network_interface.h"
struct FakeTcp : Tcp {
    std::thread rx;std::atomic<bool> stop{false};std::atomic<bool> joined{false};
    std::function<void()> before_join;
    std::function<int(FakeTcp&,const std::string&)> on_send;
    bool Connect(const std::string&,int) override {connected_=true;return true;}
    int Send(const std::string& s) override{return on_send?on_send(*this,s):s.size();}
    int GetLastError() override{return 0;}
    void Disconnect() override {connected_=false;stop=true;if(before_join)before_join();if(rx.joinable())rx.join();joined=true;}
    ~FakeTcp(){assert(!rx.joinable());}
    void dropped(Gate& gate){rx=std::thread([&]{connected_=false;disconnect_callback_();gate.block();});}
    void stream(const std::string& s){stream_callback_(s);}
};
struct Network : NetworkInterface {
    FakeTcp* last=nullptr;
    int created=0;
    std::function<int(FakeTcp&,const std::string&)> on_send;
    std::unique_ptr<Tcp> CreateTcp(int) override {++created;auto t=std::make_unique<FakeTcp>();last=t.get();t->on_send=on_send;return t;}
    std::unique_ptr<Tcp> CreateSsl(int i) override{return CreateTcp(i);}
};
int main(){
  { Network net;create_event_ok=false;{HttpClient http(&net);assert(!http.Open("GET","https://test/"));assert(http.GetLastError()==ENOMEM);}create_event_ok=true;
  } puts("PASS HTTP: event allocation failure rejects Open and destructor tolerates null");
  { Network net;auto http=std::make_unique<HttpClient>(&net);assert(http->Open("GET","https://test/"));
    Gate gate;auto* transport=net.last;transport->dropped(gate);gate.wait();int events=deleted_events;std::atomic<bool> done=false;
    std::thread owner([&]{http.reset();done=true;});until([&]{return transport->stop.load();});
    assert(!done&&deleted_events==events);gate.release();owner.join();assert(done&&deleted_events==events+1);
  } puts("PASS HTTP: already-disconnected destructor joins before deleting callback resources");
  { Network net;HttpClient http(&net);assert(http.Open("GET","https://test/one"));
    net.last->stream("HTTP/1.1 201 Created\r\nContent-Length: 1\r\n\r\nx");Gate gate;auto* old=net.last;old->dropped(gate);gate.wait();
    // The real GetStatusCode exposes parser state: reset must happen after join.
    old->before_join=[&]{assert(http.GetStatusCode()==201);};
    auto opened=std::async(std::launch::async,[&]{return http.Open("GET","https://test/two");});until([&]{return old->stop.load();});
    assert(opened.wait_for(10ms)==std::future_status::timeout);gate.release();assert(opened.get());http.Close();
  } puts("PASS HTTP: replacement joins the old parser before ResetRequestState");
  { Network net;HttpClient http(&net);assert(http.Open("GET","https://test/"));auto* transport=net.last;
    transport->stream("HTTP/1.1 200 OK\r\nContent-Length: 20000\r\n\r\n"+std::string(6000,'x'));
    std::atomic<bool> entered=false,finished=false;
    transport->rx=std::thread([&]{entered=true;transport->stream(std::string(3000,'x'));finished=true;});
    until([&]{return entered.load();});std::this_thread::sleep_for(15ms);assert(!finished);
    http.Close();assert(finished&&transport->joined);
  } puts("PASS HTTP: Close wakes a backpressured receive callback before joining");
  { Network net;HttpClient http(&net);assert(http.Open("GET","https://test/one"));auto* transport=net.last;
    transport->stream("HTTP/1.1 200 OK\r\nContent-Length: 20000\r\n\r\n"+std::string(6000,'x'));
    std::atomic<bool> entered=false,finished=false;
    transport->rx=std::thread([&]{entered=true;transport->stream(std::string(3000,'x'));finished=true;});
    until([&]{return entered.load();});std::this_thread::sleep_for(15ms);assert(!finished);
    assert(http.Open("GET","https://test/two"));assert(finished&&net.created==2);http.Close();
  } puts("PASS HTTP: replacement does not wait on a backpressured parser before Close");
  { Network net;std::atomic<bool> entered=false,finished=false;
    net.on_send=[&](FakeTcp& tcp,const std::string&){
      tcp.stream("HTTP/1.1 200 OK\r\nContent-Length: 20000\r\n\r\n"+std::string(6000,'x'));
      tcp.rx=std::thread([&]{entered=true;tcp.stream(std::string(3000,'x'));finished=true;});
      until([&]{return entered.load();});std::this_thread::sleep_for(15ms);assert(!finished);return -1;
    };
    HttpClient http(&net);assert(!http.Open("GET","https://test/"));assert(finished&&net.last->joined);
  } puts("PASS HTTP: failed request send wakes backpressured RX before joining");
  { Network net;HttpClient http(&net);http.SetKeepAlive(true);assert(http.Open("GET","https://one:443/first"));
    auto complete=[&]{net.last->stream("HTTP/1.1 200 OK\r\nContent-Length: 1\r\nConnection: keep-alive\r\n\r\nx");};
    complete();assert(http.Open("GET","https://one:443/second"));assert(net.created==1);
    complete();assert(http.Open("GET","https://two:443/"));assert(net.created==2);
    complete();assert(http.Open("GET","https://two:444/"));assert(net.created==3);
    complete();assert(http.Open("GET","http://two:444/"));assert(net.created==4);http.Close();
  } puts("PASS HTTP: keepalive reuses only the same host, port and transport scheme");
  { Network net;HttpClient http(&net);assert(http.Open("GET","https://test/"));
    assert(http.GetResponseHeader("Content-Length").empty());
    net.last->rx=std::thread([&]{std::this_thread::sleep_for(15ms);
      net.last->stream("HTTP/1.1 202 Accepted\r\nContent-Length: 3\r\nX-Test: ready\r\n\r\nabc");});
    assert(http.GetStatusCode()==202);assert(http.GetResponseHeader("x-test")=="ready");assert(http.GetBodyLength()==3);http.Close();
  } puts("PASS HTTP: async header publication synchronizes owner metadata reads");
}
'''


def main():
    with tempfile.TemporaryDirectory(prefix="puppy-ssl-host-") as directory:
        temp = Path(directory)
        headers = {"runtime.h": RUNTIME, "esp_tls.h": TLS, "network_interface.h": NETWORK}
        for name in ("esp_log.h", "esp_crt_bundle.h", "esp_timer.h", "esp_heap_caps.h",
                     "freertos/FreeRTOS.h", "freertos/event_groups.h", "freertos/task.h"):
            headers[name] = '#include "runtime.h"\n'
        for name, content in headers.items():
            path = temp / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        compiler = shlex.split(os.environ.get("CXX", "clang++"))
        sanitizers = os.environ.get("PUPPY_HOST_SANITIZERS", "address,undefined")
        if sanitizers not in ("address,undefined", "thread"):
            raise ValueError("PUPPY_HOST_SANITIZERS must be address,undefined or thread")
        common = ["-std=c++17", "-pthread", "-g", "-O1", "-fsanitize=" + sanitizers,
                  "-fno-omit-frame-pointer", "-I" + str(temp), "-I" + str(COMPONENT / "include"),
                  "-I" + str(COMPONENT / "src/esp")]
        for name, test, source, variants in (
            ("ssl", SSL_TEST, COMPONENT / "src/esp/esp_ssl.cc", (0, 1)),
            ("http", HTTP_TEST, COMPONENT / "src/http_client.cc", (0,)),
        ):
            path = temp / (name + "_test.cc")
            path.write_text(test)
            for external_stack in variants:
                output = temp / f"{name}-{external_stack}"
                subprocess.run(compiler + common + [f"-DCONFIG_IDF_TARGET_ESP32S3={external_stack}",
                    f"-DCONFIG_SPIRAM_ALLOW_STACK_EXTERNAL_MEMORY={external_stack}", str(source), str(path), "-o", str(output)], check=True)
                subprocess.run([str(output)], check=True, timeout=30)


if __name__ == "__main__":
    main()
