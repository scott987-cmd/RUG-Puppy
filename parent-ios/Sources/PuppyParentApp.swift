import SwiftUI
import UniformTypeIdentifiers

@main
struct PuppyParentApp: App {
    @StateObject private var model = ParentModel()
    var body: some Scene {
        WindowGroup { ParentView().environmentObject(model) }
    }
}

private let ink = Color(red: 0.12, green: 0.20, blue: 0.23)
private let teal = Color(red: 0.10, green: 0.43, blue: 0.40)
private let canvas = Color(red: 0.96, green: 0.96, blue: 0.93)

struct ParentView: View {
    @EnvironmentObject var model: ParentModel
    @State private var selectedTab = 0
    var body: some View {
        TabView(selection: $selectedTab) {
            Dashboard(onOpenFamily: { selectedTab = 1 }).tabItem { Label("小狗", systemImage: "pawprint.fill") }.tag(0)
            NativeFamilyView().tabItem { Label("家人", systemImage: "person.2.crop.square.stack") }.tag(1)
            ManagementHomeView().tabItem { Label("管理", systemImage: "slider.horizontal.3") }.tag(2)
            HistoryView().tabItem { Label("记录", systemImage: "clock.arrow.circlepath") }.tag(3)
            ConnectionView().tabItem { Label("连接", systemImage: "link") }.tag(4)
        }
        .tint(teal)
        .task {
            while !Task.isCancelled {
                await model.refresh()
                if model.shouldCheckMobileAdmin { await model.checkMobileAdmin() }
                try? await Task.sleep(for: .seconds(10))
            }
        }
    }
}

private struct DemoHost: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var model = ParentModel(demo: true)

    var body: some View {
        ParentView()
            .environmentObject(model)
            .safeAreaInset(edge: .top) {
                HStack {
                    Label("离线演示 · 不连接家中设备", systemImage: "testtube.2")
                        .font(.caption.weight(.semibold))
                    Spacer()
                    Button("退出演示") { dismiss() }
                        .accessibilityIdentifier("exitDemoButton")
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(canvas)
            }
    }
}

struct Dashboard: View {
    @EnvironmentObject var model: ParentModel
    let onOpenFamily: () -> Void
    @State private var activeButtonID: String?
    @State private var directActionPending = false
    @State private var actionRouteText = "检测中…"
    @State private var manualRefreshing = false
    @State private var feedbackGroup: String?
    @State private var feedbackText: String?
    @State private var feedbackRequestID: String?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    if model.isDemo {
                        Label("演示 · 隔离数据，不连接真实小狗", systemImage: "testtube.2")
                            .font(.caption).foregroundStyle(teal).accessibilityIdentifier("demoBanner")
                    }
                    TimelineView(.periodic(from: .now, by: 1)) { timeline in
                        hero(now: timeline.date)
                    }
                    Button(action: onOpenFamily) {
                        HStack(spacing: 12) {
                            Image(systemName: "waveform.badge.mic")
                                .font(.title2)
                            VStack(alignment: .leading, spacing: 3) {
                                Text("给家人录入声纹").font(.headline)
                                Text("用手机麦克风录入，在「家人」管理")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            Image(systemName: "chevron.right").font(.caption.weight(.semibold))
                        }
                        .foregroundStyle(ink)
                    }
                    .accessibilityIdentifier("openVoiceEnrollmentButton")
                    .card()
                    if !model.ready {
                        Label("到「连接」导入家庭配置并选择 iCloud 目录", systemImage: "arrow.down.right")
                            .font(.subheadline).foregroundStyle(.secondary)
                    }
                    VStack(alignment: .leading, spacing: 12) {
                        Text("看看家里的小狗").font(.headline)
                        Label(model.isDemo ? "当前通道：演示模式" : "当前通道：iCloud 异步",
                              systemImage: model.isDemo ? "testtube.2" : "icloud")
                            .font(.caption.weight(.medium)).foregroundStyle(.secondary)
                            .accessibilityIdentifier("observeRouteLabel")
                        HStack(spacing: 12) {
                            requestButton("查看状态", icon: "arrow.triangle.2.circlepath", kind: .status, id: "statusButton")
                            requestButton("拍张照片", icon: "camera", kind: .photo, id: "photoButton")
                        }
                        HStack(spacing: 12) {
                            requestButton("定点巡看", icon: "eye", kind: .patrolOnce, id: "patrolButton")
                            requestButton("请求停止", icon: "stop.circle", kind: .stop, id: "stopButton")
                        }
                        commandFeedback(for: "observe")
                        Text("巡看只在当前位置拍一张照片。远程照片会经你的 iCloud Drive 同步到手机；照片识别和看图对话在家中 Mac 处理。")
                            .font(.caption).foregroundStyle(.secondary)
                    }.card()
                    VStack(alignment: .leading, spacing: 12) {
                        Text("打个招呼").font(.headline)
                        Label("当前动作通道：\(actionRouteText)",
                              systemImage: actionRouteText == "家中局域网直连" ? "wifi" : "icloud")
                            .font(.caption.weight(.medium)).foregroundStyle(.secondary)
                            .accessibilityIdentifier("actionRouteLabel")
                        HStack(spacing: 10) {
                            ForEach([("wave", "挥手"), ("sit", "坐下"), ("stretch", "伸展"), ("hug", "抱抱")], id: \.0) { action in
                                Button { sendHomeCommand(.action, action: action.0, id: "action-\(action.0)", group: "action") } label: {
                                    buttonLabel(action.1, icon: nil, id: "action-\(action.0)")
                                }
                                .buttonStyle(.bordered).frame(maxWidth: .infinity)
                                .disabled((!model.ready && model.pairing?.adminURL == nil) ||
                                          activeButtonID != nil || directActionPending || model.sending || model.isDemo)
                                .accessibilityIdentifier("action-\(action.0)")
                            }
                        }
                        commandFeedback(for: "action")
                        Text("请确认周围安全。本机按键用于急停，远程停止不能保证及时到达。")
                            .font(.caption).foregroundStyle(.secondary)
                    }.card()
                    if let record = model.records.first(where: { model.photos[$0.id] != nil }), let image = model.photos[record.id] {
                        VStack(alignment: .leading, spacing: 12) {
                            HStack { Text("最近的照片").font(.headline); Spacer(); Image(systemName: "checkmark.shield").foregroundStyle(teal) }
                            Image(uiImage: image).resizable().scaledToFit().frame(maxWidth: .infinity, maxHeight: 280)
                                .clipShape(RoundedRectangle(cornerRadius: 18)).accessibilityIdentifier("verifiedPhoto")
                            if let captured = record.receipt?.result?.capturedAt {
                                Text("拍摄于 \(Date(timeIntervalSince1970: captured).formatted(date: .abbreviated, time: .standard))")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            Text("已验证服务器签名和照片完整性").font(.caption2).foregroundStyle(teal)
                            PhotoDescriptionView(description: record.receipt?.result?.description)
                        }.card()
                    }
                    if let message = model.message { Text(message).font(.footnote).foregroundStyle(.secondary).accessibilityIdentifier("statusMessage") }
                }.padding(20).frame(maxWidth: 720)
            }
            .frame(maxWidth: .infinity).background(canvas).navigationTitle("Puppy")
            .toolbar {
                Button { refreshFromToolbar() } label: {
                    if manualRefreshing { ProgressView().controlSize(.small) }
                    else { Image(systemName: "arrow.clockwise") }
                }
                .disabled(!model.ready || manualRefreshing || model.refreshing || model.sending)
                .accessibilityIdentifier("homeRefreshButton")
            }
            .refreshable { await model.refresh() }
            .task(id: model.pairing?.adminURL) {
                while !Task.isCancelled {
                    await checkActionRoute()
                    try? await Task.sleep(for: .seconds(20))
                }
            }
        }
    }

    private func hero(now: Date) -> some View {
        let online = model.snapshot?.online(at: now) == true
        let stale = model.snapshot != nil && !(model.snapshot?.status.fresh(at: now) ?? false)
        return HStack(spacing: 18) {
            Text("🐶").font(.system(size: 62)).padding(10).background(.white.opacity(0.6), in: RoundedRectangle(cornerRadius: 24))
            VStack(alignment: .leading, spacing: 9) {
                Text(model.snapshot?.status.petName ?? "家里的小狗").font(.title.bold()).foregroundStyle(ink)
                Label(online ? "在线" : stale ? "状态已过期" : "尚未确认在线", systemImage: online ? "circle.fill" : "circle")
                    .font(.subheadline.weight(.medium)).foregroundStyle(online ? teal : .secondary)
                    .accessibilityIdentifier("onlineState")
                HStack(spacing: 14) {
                    if let volume = model.snapshot?.status.volume { Label(volume == 0 ? "已静音" : "音量 \(volume)", systemImage: volume == 0 ? "speaker.slash" : "speaker.wave.1") }
                    if let battery = model.snapshot?.status.battery?.level { Label("电量估计 \(battery)%", systemImage: "battery.75percent") }
                }.font(.caption).foregroundStyle(.secondary)
                if let battery = model.snapshot?.status.battery, battery.chargingKnown == false {
                    if let voltage = battery.servoVoltageV {
                        Text("舵机供电 \(voltage, specifier: "%.1f") V · 充电状态未检测")
                            .font(.caption2).foregroundStyle(.secondary)
                    } else {
                        Text("充电状态未检测").font(.caption2).foregroundStyle(.secondary)
                    }
                }
                if let observed = model.snapshot?.status.observedAt {
                    Text("设备观察于 \(Date(timeIntervalSince1970: observed).formatted(date: .omitted, time: .standard))")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
            Spacer(minLength: 0)
        }.padding(18).frame(maxWidth: .infinity).background(Color(red: 0.86, green: 0.93, blue: 0.88), in: RoundedRectangle(cornerRadius: 28))
    }

    private func requestButton(_ title: String, icon: String, kind: CommandKind, id: String) -> some View {
        Button { sendHomeCommand(kind, id: id, group: "observe") } label: {
            buttonLabel(title, icon: icon, id: id)
        }
        .buttonStyle(.bordered).tint(kind == .stop ? .orange : teal)
        .disabled(!model.ready || activeButtonID != nil || model.sending || model.refreshing || (model.isDemo && kind != .status))
        .accessibilityIdentifier(id)
    }

    private func buttonLabel(_ title: String, icon: String?, id: String) -> some View {
        HStack(spacing: 6) {
            if activeButtonID == id { ProgressView().controlSize(.small) }
            else if let icon { Image(systemName: icon) }
            Text(activeButtonID == id ? "发送中…" : title)
        }
        .font(.subheadline.weight(.semibold))
        .frame(maxWidth: .infinity, minHeight: 46)
    }

    private func sendHomeCommand(_ kind: CommandKind, action: String? = nil, id: String, group: String) {
        guard activeButtonID == nil else { return }
        activeButtonID = id
        feedbackGroup = group
        feedbackText = nil
        feedbackRequestID = nil
        Task {
            let started = Date()
            let handledDirectly: Bool
            if kind == .action, let action {
                handledDirectly = await sendLocalActionIfAvailable(action)
            } else {
                handledDirectly = false
            }
            if !handledDirectly {
                let previousID = model.records.first?.id
                await model.send(kind, action: action)
                if model.records.first?.id != previousID { feedbackRequestID = model.records.first?.id }
                feedbackText = "本次通道：iCloud 异步。" +
                    (model.message ?? "请求已结束，请到「记录」查看")
            }
            let remaining = 0.45 - Date().timeIntervalSince(started)
            if remaining > 0 { try? await Task.sleep(for: .seconds(remaining)) }
            activeButtonID = nil
        }
    }

    private func sendLocalActionIfAvailable(_ action: String) async -> Bool {
        guard !model.isDemo, let pairing = model.pairing, pairing.adminURL != nil else { return false }
        do {
            let client = try model.mobileClient()
            do {
                // A read-only probe can safely fall back to iCloud if the phone is away from home.
                _ = try await client.json("GET", "/api/status", timeout: 2.5)
            } catch AdminError.network {
                actionRouteText = "iCloud 异步"
                return false
            } catch {
                actionRouteText = "本地连接异常"
                feedbackText = "家庭连接检查失败，动作未提交：\(error.localizedDescription)"
                return true
            }
            actionRouteText = "家中局域网直连"
            let result = try await client.json("POST", "/api/devices/\(pairing.deviceID)/action/\(action)",
                                               timeout: 8)
            guard result["accepted"] as? Bool == true else { throw AdminError.invalidResponse }
            feedbackText = "已通过局域网送达家中 Mac，正在等待小陆完成动作"
            directActionPending = true
            Task { await watchLocalAction(action, deviceID: pairing.deviceID) }
        } catch AdminError.server(_, let detail) {
            feedbackText = "动作未提交：\(detail)"
        } catch AdminError.trust {
            feedbackText = "动作未提交：家庭服务器证书校验失败"
        } catch {
            feedbackText = "动作提交结果未确认：\(error.localizedDescription)。请先看小陆状态，不要重复点击"
        }
        return true
    }

    private func checkActionRoute() async {
        if model.isDemo { actionRouteText = "演示模式"; return }
        guard model.pairing?.adminURL != nil else { actionRouteText = "iCloud 异步"; return }
        do {
            _ = try await model.mobileClient().json("GET", "/api/status", timeout: 2.5)
            if !Task.isCancelled { actionRouteText = "家中局域网直连" }
        } catch AdminError.network {
            if !Task.isCancelled { actionRouteText = "iCloud 异步" }
        } catch {
            if !Task.isCancelled { actionRouteText = "本地连接异常" }
        }
    }

    private func watchLocalAction(_ action: String, deviceID: String) async {
        defer { directActionPending = false }
        let deadline = Date().addingTimeInterval(12)
        while Date() < deadline {
            try? await Task.sleep(for: .seconds(1))
            guard !Task.isCancelled else { return }
            do {
                let status = try await model.mobileClient().json("GET", "/api/status", timeout: 2.5)
                guard let actions = status["actions"] as? [String: Any],
                      let job = actions[deviceID] as? [String: Any],
                      job["name"] as? String == action else { continue }
                switch job["state"] as? String {
                case "completed": feedbackText = "小陆已完成\(actionTitle(action))"; return
                case "failed", "cancelled":
                    feedbackText = "动作未完成：\(job["error"] as? String ?? "请查看小陆状态")"
                    return
                default: break
                }
            } catch { continue }
        }
        feedbackText = "请求已送达，但完成状态暂未确认；请先看小陆状态，不要重复点击"
    }

    private func actionTitle(_ action: String) -> String {
        ["wave": "挥手", "sit": "坐下", "stretch": "伸展", "hug": "抱抱"][action] ?? "动作"
    }

    @ViewBuilder
    private func commandFeedback(for group: String) -> some View {
        if feedbackGroup == group, let feedbackText {
            if let feedbackRequestID,
               let record = model.records.first(where: { $0.id == feedbackRequestID }),
               let receipt = record.receipt {
                Label("\(record.command.kind.title)：\(receipt.state.title)",
                      systemImage: receipt.state == .succeeded ? "checkmark.circle" : "clock")
                    .font(.footnote).foregroundStyle(.secondary)
                    .accessibilityIdentifier("homeCommandFeedback")
            } else {
                Text(feedbackText).font(.footnote).foregroundStyle(.secondary)
                    .accessibilityIdentifier("homeCommandFeedback")
            }
        }
    }

    private func refreshFromToolbar() {
        guard !manualRefreshing, !model.refreshing else { return }
        manualRefreshing = true
        feedbackGroup = "observe"
        feedbackText = nil
        feedbackRequestID = nil
        Task {
            let started = Date()
            let previousRefresh = model.lastRefresh
            await model.refresh()
            feedbackText = model.lastRefresh != previousRefresh ? "已检查家中状态" : (model.message ?? "刷新未完成")
            let remaining = 0.45 - Date().timeIntervalSince(started)
            if remaining > 0 { try? await Task.sleep(for: .seconds(remaining)) }
            manualRefreshing = false
        }
    }
}

struct HistoryView: View {
    @EnvironmentObject var model: ParentModel
    var body: some View {
        NavigationStack {
            Group {
                if model.records.isEmpty {
                    ContentUnavailableView("还没有请求", systemImage: "clock", description: Text("请求提交后，会在这里显示家中 Mac 的实际回执。"))
                } else {
                    TimelineView(.periodic(from: .now, by: 1)) { timeline in
                        List(model.records) { record in
                            VStack(alignment: .leading, spacing: 8) {
                                HStack { Text(record.command.kind.title).font(.headline); Spacer(); Text(record.title(now: timeline.date)).font(.subheadline).foregroundStyle(teal) }
                                Text(Date(timeIntervalSince1970: Double(record.command.createdAt)).formatted(date: .abbreviated, time: .standard)).font(.caption).foregroundStyle(.secondary)
                                if record.receipt?.state.terminal != true {
                                    Text("提交不等于完成。有效期结束后仍以服务端回执为准。")
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                                if record.receipt?.state == .unknown {
                                    Text("可能已执行，请先确认小狗状态；不会自动重发。").font(.caption).foregroundStyle(.secondary)
                                }
                                if record.receipt == nil, let notice = record.rejectionNotice {
                                    Text("收到同 ID 文件的拒绝通知（\(notice.error ?? "rejected")）。该通知不确定对应本次请求，仍等待正式回执。")
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                                if let reason = record.receipt?.error ?? record.receipt?.result?.error {
                                    Text(reason == "expired" ? "到达时已超过有效期" : "服务端原因：\(reason)").font(.caption).foregroundStyle(.secondary)
                                }
                                if [.photo, .patrolOnce].contains(record.command.kind) && record.receipt?.state == .succeeded {
                                    if record.receipt?.result?.media != nil && model.photos[record.id] == nil {
                                        Text(model.unavailablePhotos.contains(record.id)
                                             ? "照片文件暂不可用，可能仍在同步或已按保留期清理。"
                                             : "照片正在同步，请稍后刷新。")
                                            .font(.caption).foregroundStyle(.secondary)
                                            .accessibilityIdentifier("photoUnavailable")
                                    }
                                    PhotoDescriptionView(description: record.receipt?.result?.description,
                                                         photoAvailable: model.photos[record.id] != nil)
                                }
                                Text(String(record.id.prefix(8))).font(.caption2.monospaced()).foregroundStyle(.tertiary)
                            }.padding(.vertical, 8)
                        }
                    }
                }
            }.navigationTitle("请求记录").refreshable { await model.refresh() }
        }
    }
}

struct PhotoDescriptionView: View {
    let description: PhotoDescription?
    let photoAvailable: Bool
    init(description: PhotoDescription?, photoAvailable: Bool = true) {
        self.description = description
        self.photoAvailable = photoAvailable
    }
    var body: some View {
        if let reply = description?.displayReply {
            VStack(alignment: .leading, spacing: 6) {
                Label("单张照片摘要", systemImage: "text.bubble").font(.subheadline.weight(.medium))
                Text(reply).font(.subheadline).accessibilityIdentifier("photoDescriptionReply")
                if description?.uncertain == true {
                    Text("画面信息有限，描述可能不完整。").font(.caption).foregroundStyle(.secondary)
                }
                Text("仅针对这张照片，不代表整个房间或实时情况。")
                    .font(.caption).foregroundStyle(.secondary).accessibilityIdentifier("photoDescriptionScope")
            }
        } else if description?.ok == false || description?.error != nil {
            Text(photoAvailable ? "照片可查看，画面摘要暂不可用。" : "照片与画面摘要暂不可用。")
                .font(.caption).foregroundStyle(.secondary).accessibilityIdentifier("photoDescriptionUnavailable")
        }
    }
}

struct ConnectionView: View {
    @EnvironmentObject var model: ParentModel
    @State private var importConfiguration = false
    @State private var chooseDirectory = false
    @State private var copied = false
    @State private var showDemo = false
    @State private var adminChecking = false
    @State private var adminMessage = ""
    @State private var statusRequestMessage = ""
    var body: some View {
        NavigationStack {
            Form {
                if !model.isDemo {
                    Section("先看看") {
                        Button("体验离线演示") { showDemo = true }
                            .accessibilityIdentifier("enterDemoButton")
                        Text("使用本机生成的示例状态和照片；不会连接小狗，也不会驱动电机。")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                Section("1 · 家庭服务器") {
                    Button("导入 Mac 配对配置") { importConfiguration = true }
                        .disabled(model.isDemo)
                        .accessibilityIdentifier("importPairingButton")
                        .fileImporter(isPresented: $importConfiguration, allowedContentTypes: [.json]) { result in
                            if case .success(let url) = result { Task { await model.importPairing(url) } }
                            else if case .failure(let error) = result { model.message = error.localizedDescription }
                        }
                    if let pairing = model.pairing {
                        LabeledContent("服务器", value: pairing.serverID)
                        LabeledContent("Puppy", value: pairing.deviceID)
                        LabeledContent("公钥指纹", value: pairing.fingerprint).font(.caption.monospaced())
                        Text("请与家中 Mac 显示的指纹核对。配置只包含公钥，不包含服务器私钥。").font(.caption).foregroundStyle(.secondary)
                    }
                }
                Section("2 · iCloud 共享目录") {
                    Button("选择 PuppyCompanion 目录") { chooseDirectory = true }
                        .disabled(model.isDemo)
                        .accessibilityIdentifier("chooseDirectoryButton")
                        .fileImporter(isPresented: $chooseDirectory, allowedContentTypes: [.folder]) { result in
                            if case .success(let url) = result { Task { await model.selectDirectory(url) } }
                            else if case .failure(let error) = result { model.message = error.localizedDescription }
                        }
                    if model.directory != nil { Label("目录已选择", systemImage: "folder.badge.checkmark") }
                    Text("在「文件」中选择你自己的 PuppyCompanion 目录。小狗不登录 iCloud。").font(.caption).foregroundStyle(.secondary)
                }
                Section("3 · 登记这台手机") {
                    if let identity = model.identity {
                        Text(identity.sourceID).font(.caption.monospaced()).textSelection(.enabled)
                        Button(copied ? "已复制公钥登记信息" : "复制手机公钥登记信息") {
                            UIPasteboard.general.string = model.registration
                            copied = true
                        }.disabled(model.pairing == nil || model.isDemo).accessibilityIdentifier("copyPublicKeyButton")
                        ShareLink("分享公钥登记信息", item: model.registration).disabled(model.pairing == nil || model.isDemo)
                        Text("在家中 Mac 登记后才能发送命令。手机私钥只保存在本机钥匙串，复制和分享仅含公钥。").font(.caption).foregroundStyle(.secondary)
                    }
                }
                Section("家庭管理连接") {
                    if let address = model.pairing?.adminURL {
                        Label("已由家庭服务器签名", systemImage: "checkmark.shield")
                            .foregroundStyle(teal)
                        Text(address).font(.caption.monospaced()).textSelection(.enabled)
                        Button(adminChecking ? "正在检查…" : "检查家中管理连接") {
                            adminChecking = true
                            adminMessage = "正在检查手机与家中 Mac 的连接…"
                            Task {
                                defer { adminChecking = false }
                                do {
                                    _ = try await model.mobileClient().json("GET", "/api/status")
                                    adminMessage = "已通过手机签名连接家中 Mac"
                                } catch { adminMessage = error.localizedDescription }
                            }
                        }
                        .disabled(adminChecking || model.isDemo)
                        .accessibilityIdentifier("checkMobileAdminButton")
                    } else {
                        Text("等待 iCloud 同步服务器签名的手机连接配置。")
                            .font(.caption).foregroundStyle(.secondary)
                        Button("刷新配对信息") {
                            adminMessage = "正在从 iCloud 读取家庭配对信息…"
                            Task {
                                await model.refresh()
                                adminMessage = model.pairing?.adminURL == nil
                                    ? (model.message ?? "尚未收到手机管理配置，请稍后重试")
                                    : "已收到家庭管理配置，现在可以检查连接"
                            }
                        }
                            .disabled(!model.ready || model.refreshing)
                    }
                    if !adminMessage.isEmpty {
                        HStack {
                            if adminChecking { ProgressView().controlSize(.small) }
                            Text(adminMessage)
                        }
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .accessibilityIdentifier("mobileAdminCheckResult")
                    }
                    else if let message = model.adminConnectionMessage { Text(message).font(.footnote).foregroundStyle(.secondary) }
                }
                Section {
                    Button("发送状态请求，检查连接") {
                        statusRequestMessage = "正在向 iCloud 提交状态请求…"
                        Task {
                            await model.send(.status)
                            statusRequestMessage = model.message ?? "请在「记录」查看回执"
                        }
                    }
                        .disabled(!model.ready || model.sending || model.refreshing).accessibilityIdentifier("testConnectionButton")
                    if !statusRequestMessage.isEmpty {
                        Text(statusRequestMessage).font(.footnote).foregroundStyle(.secondary)
                            .accessibilityIdentifier("statusRequestResult")
                    }
                    Text("后台收到回执后才能确认连接成功。手机应用关闭时不会保证持续同步。").font(.caption).foregroundStyle(.secondary)
                    if model.isDemo { Text("当前为隔离演示，无法连接真实目录。").foregroundStyle(.orange) }
                }
            }
            .navigationTitle("连接家庭")
        }
        .fullScreenCover(isPresented: $showDemo) { DemoHost() }
    }
}

private extension View {
    func card() -> some View { padding(18).frame(maxWidth: .infinity, alignment: .leading).background(.white, in: RoundedRectangle(cornerRadius: 24)) }
}
