import SwiftUI
import AVFoundation
import UIKit
import Vision

private let adminTint = Color(red: 0.10, green: 0.43, blue: 0.40)

private struct EditedActionStep: Identifiable {
    let id = UUID()
    var action = "wave"
    var waitMS = 0
}

@MainActor private func devicePath(_ model: ParentModel) -> String? {
    guard let id = model.pairing?.deviceID, Pairing.validID(id) else { return nil }
    return id
}

private func safeMember(_ value: String) -> Bool {
    value.range(of: "^[A-Za-z0-9_\\-\\u4e00-\\u9fff]{1,64}$", options: .regularExpression) != nil
}

@MainActor private final class PhoneVoiceRecorder {
    private let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
    private var recorder: AVAudioRecorder?

    func start() async throws {
        let allowed = await withCheckedContinuation { continuation in
            AVAudioApplication.requestRecordPermission { continuation.resume(returning: $0) }
        }
        guard allowed else { throw AdminError.invalidInput("请在 iPhone 设置中允许 Puppy 家长使用麦克风") }
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .default)
        try session.setActive(true)
        if let builtIn = session.availableInputs?.first(where: { $0.portType == .builtInMic }) {
            try session.setPreferredInput(builtIn)
        }
        let settings: [String: Any] = [AVFormatIDKey: Int(kAudioFormatLinearPCM), AVSampleRateKey: 16_000,
                                       AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16,
                                       AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false]
        let recorder = try AVAudioRecorder(url: url, settings: settings)
        recorder.isMeteringEnabled = true
        guard recorder.prepareToRecord(), recorder.record(forDuration: 20) else {
            try? session.setActive(false)
            throw AdminError.invalidInput("手机麦克风未能开始录音")
        }
        self.recorder = recorder
    }

    func inputLevel() -> Float? {
        guard let recorder, recorder.isRecording else { return nil }
        recorder.updateMeters()
        return recorder.averagePower(forChannel: 0)
    }

    func finish() throws -> Data {
        guard let recorder else { throw AdminError.invalidInput("录音尚未开始") }
        recorder.stop()
        self.recorder = nil
        defer { try? FileManager.default.removeItem(at: url) }
        try? AVAudioSession.sharedInstance().setActive(false)
        let wav = try Data(contentsOf: url)
        guard (96_000...644_096).contains(wav.count) else {
            throw AdminError.invalidInput("请录制 3–20 秒后再保存")
        }
        return wav
    }

    func cancel() {
        recorder?.stop()
        recorder = nil
        try? AVAudioSession.sharedInstance().setActive(false)
        try? FileManager.default.removeItem(at: url)
    }
}

private struct PhoneCameraCapture: UIViewControllerRepresentable {
    let onCapture: (UIImage?) -> Void

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.cameraCaptureMode = .photo
        if UIImagePickerController.isCameraDeviceAvailable(.front) {
            picker.cameraDevice = .front
        }
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ controller: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(onCapture: onCapture) }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let onCapture: (UIImage?) -> Void
        init(onCapture: @escaping (UIImage?) -> Void) { self.onCapture = onCapture }
        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            onCapture(info[.originalImage] as? UIImage)
        }
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) { onCapture(nil) }
    }
}

@MainActor private func findPhoneFace(_ image: UIImage) throws -> VNFaceObservation? {
    guard let pixels = image.cgImage else { return nil }
    let request = VNDetectFaceRectanglesRequest()
    try VNImageRequestHandler(cgImage: pixels).perform([request])
    let faces = request.results ?? []
    guard faces.count <= 1 else {
        throw AdminError.invalidInput("照片里有多张人脸，请只拍要录入的本人")
    }
    return faces.first
}

@MainActor private func faceJPEG(_ image: UIImage) throws -> Data {
    guard min(image.size.width, image.size.height) >= 64 else {
        throw AdminError.invalidInput("照片尺寸太小，请重新拍摄")
    }
    let ratio = min(1.0, 1200 / max(image.size.width, image.size.height))
    let size = CGSize(width: (image.size.width * ratio).rounded(),
                      height: (image.size.height * ratio).rounded())
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    format.opaque = true
    let normalized = UIGraphicsImageRenderer(size: size, format: format).image { _ in
        image.draw(in: CGRect(origin: .zero, size: size))
    }
    guard let normalizedPixels = normalized.cgImage else {
        throw AdminError.invalidInput("照片无法处理，请重新拍摄")
    }
    var candidates = [normalized]
    for orientation in [UIImage.Orientation.left, .right, .down] {
        let rotated = UIImage(cgImage: normalizedPixels, scale: 1, orientation: orientation)
        let output = CGSize(width: orientation == .down ? size.width : size.height,
                            height: orientation == .down ? size.height : size.width)
        candidates.append(UIGraphicsImageRenderer(size: output, format: format).image { _ in
            rotated.draw(in: CGRect(origin: .zero, size: output))
        })
    }
    var selected: (UIImage, VNFaceObservation)?
    for candidate in candidates {
        if let face = try findPhoneFace(candidate) { selected = (candidate, face); break }
    }
    if selected == nil {
        for candidate in candidates {
            guard let pixels = candidate.cgImage else { continue }
            let width = CGFloat(pixels.width), height = CGFloat(pixels.height)
            let side = min(width, height) * 0.78
            for centerY in [height * 0.40, height * 0.55] {
                let area = CGRect(x: (width - side) / 2, y: centerY - side / 2,
                                  width: side, height: side).integral
                guard let crop = pixels.cropping(to: area) else { continue }
                let closer = UIImage(cgImage: crop)
                if let face = try findPhoneFace(closer) {
                    selected = (closer, face)
                    break
                }
            }
            if selected != nil { break }
        }
    }
    guard let (selectedImage, face) = selected, let pixels = selectedImage.cgImage else {
        throw AdminError.invalidInput("手机未检测到人脸。请让本人正脸占据画面中央约三分之一，避开逆光后重拍")
    }
    let box = face.boundingBox
    let width = CGFloat(pixels.width), height = CGFloat(pixels.height)
    let side = max(box.width * width, box.height * height) * 2.6
    let center = CGPoint(x: box.midX * width, y: (1 - box.midY) * height)
    let crop = CGRect(x: center.x - side / 2, y: center.y - side / 2,
                      width: side, height: side)
        .intersection(CGRect(x: 0, y: 0, width: width, height: height)).integral
    guard crop.width >= 64, crop.height >= 64, let cropped = pixels.cropping(to: crop) else {
        throw AdminError.invalidInput("人脸太靠近画面边缘，请完整入镜后重拍")
    }
    let photo = UIImage(cgImage: cropped)
    for dimension in [720.0, 600.0, 480.0] {
        let scale = min(1.0, dimension / max(photo.size.width, photo.size.height))
        let output = CGSize(width: max(1, (photo.size.width * scale).rounded()),
                            height: max(1, (photo.size.height * scale).rounded()))
        let resized = UIGraphicsImageRenderer(size: output, format: format).image { _ in
            photo.draw(in: CGRect(origin: .zero, size: output))
        }
        for quality in [0.78, 0.62, 0.48] {
            if let jpeg = resized.jpegData(compressionQuality: quality), jpeg.count <= 512_000 {
                return jpeg
            }
        }
    }
    throw AdminError.invalidInput("照片过大，请靠近并重新拍摄")
}

private struct FamilyMember: Identifiable {
    let id: String
    let name: String
    let faceCount: Int
    let voiceCount: Int
    let ready: Bool
    init?(_ object: [String: Any]) {
        guard let id = object["person_id"] as? String, let name = object["display_name"] as? String,
              let counts = object["counts"] as? [String: Int] else { return nil }
        self.id = id; self.name = name
        faceCount = counts["face"] ?? 0; voiceCount = counts["voice"] ?? 0
        ready = object["ready"] as? Bool ?? false
    }

    func readinessText(requiredSamples: Int) -> String {
        if ready { return "已可识别" }
        var missing: [String] = []
        if voiceCount < requiredSamples { missing.append("\(requiredSamples - voiceCount) 段声纹") }
        if faceCount < requiredSamples { missing.append("\(requiredSamples - faceCount) 张人脸照片") }
        if !missing.isEmpty { return "暂不能识别，还需" + missing.joined(separator: "、") }
        return "样本数量已够，但家中识别模型尚未就绪"
    }
}

struct NativeFamilyView: View {
    @EnvironmentObject private var model: ParentModel
    @State private var navigationPath = NavigationPath()
    @State private var pendingCreatedID: String?
    @State private var members: [FamilyMember] = []
    @State private var recognition = false
    @State private var requiredSamples = 2
    @State private var memberID = ""
    @State private var memberName = ""
    @State private var consent = false
    @State private var busy = false
    @State private var note = ""
    @State private var ticketID: String?
    @State private var pendingDelete: String?

    var body: some View {
        NavigationStack(path: $navigationPath) {
            Form {
                if model.pairing?.adminURL == nil {
                    ContentUnavailableView("等待家庭连接", systemImage: "lock.shield",
                        description: Text("在「连接」导入带手机管理地址和家庭证书的配对配置。"))
                } else {
                    Section("用手机录入声纹") {
                        if members.isEmpty {
                            Text("先在下方添加家人并确认同意，保存后会自动进入声纹录入。")
                                .foregroundStyle(.secondary)
                        }
                        ForEach(members) { member in
                            NavigationLink(value: member.id) {
                                VStack(alignment: .leading, spacing: 4) {
                                    Label("录入\(member.name)的声纹", systemImage: "waveform.badge.mic")
                                    Text("声纹 \(member.voiceCount) 段 · 人脸 \(member.faceCount) 张")
                                        .font(.caption).foregroundStyle(.secondary)
                                    Text(member.readinessText(requiredSamples: requiredSamples))
                                        .font(.caption)
                                        .foregroundStyle(member.ready ? Color.green : Color.orange)
                                }
                            }
                        }
                        Text("进入家人详情后，点「用手机麦克风录入声纹」；本人说话 3–20 秒，再点结束保存。")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Section("身份识别") {
                        Toggle("启用声纹与人脸识别", isOn: Binding(
                            get: { recognition }, set: { changeRecognition($0) }))
                            .disabled(busy)
                        if recognition && !members.contains(where: \.ready) {
                            Text("识别已开启，但还没有完成录入的家人。每人需录入至少 \(requiredSamples) 段声纹和 \(requiredSamples) 张人脸照片，小陆才会尝试认人。")
                                .font(.footnote).foregroundStyle(.orange)
                        }
                        Text("识别不确定时，小陆把说话的人视为陌生人，不读取个人记忆。关闭识别会保留样本，可逐项删除。")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Section("添加家人") {
                        TextField("成员 ID（可留空自动生成）", text: $memberID)
                            .textInputAutocapitalization(.never).autocorrectionDisabled()
                        TextField("称呼", text: $memberName)
                        Toggle("我同意在家中 Mac 保存此人的身份特征", isOn: $consent)
                        Button("保存家人并进入录入") { operate { client, _ in
                            let selectedID = memberID.isEmpty ? "family-" + String(UUID().uuidString.prefix(12)).lowercased() : memberID
                            guard safeMember(selectedID), !memberName.trimmingCharacters(in: .whitespaces).isEmpty,
                                  memberName.count <= 32, consent else { throw AdminError.invalidInput("请填写有效成员资料并确认同意") }
                            _ = try await client.json("PUT", "/api/identity/members/\(selectedID)",
                                                       body: ["display_name": memberName.trimmingCharacters(in: .whitespaces)])
                            memberID = selectedID
                            pendingCreatedID = selectedID
                            return "家人已保存；请录入不同角度的两张人脸和两段语音"
                        } }
                        .disabled(busy || !consent || (!memberID.isEmpty && !safeMember(memberID)) || memberName.isEmpty)
                        if !consent || memberName.isEmpty {
                            Text("先填写称呼并确认同意，成员 ID 可留空。保存后会直接打开声纹与人脸录入。")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    if !note.isEmpty { Section { Text(note).font(.footnote).foregroundStyle(.secondary) } }
                }
            }
            .navigationTitle("家人")
            .navigationDestination(for: String.self) { id in
                if let member = members.first(where: { $0.id == id }) {
                    MemberDetailView(member: member, requiredSamples: requiredSamples,
                                     onChanged: { await refresh() }).environmentObject(model)
                }
            }
            .toolbar { Button { Task { await refresh() } } label: { Image(systemName: "arrow.clockwise") }.disabled(busy) }
            .task { await refresh() }
            .refreshable { await refresh() }
        }
    }

    private func changeRecognition(_ enabled: Bool) {
        operate { client, id in
            let result = try await client.json("PUT", "/api/identity/devices/\(id)/settings", body: ["enabled": enabled])
            recognition = result["enabled"] as? Bool ?? false
            return enabled ? "身份识别已开启" : "身份识别已关闭；样本仍保留"
        }
    }

    private func operate(_ action: @escaping (MobileAdminClient, String) async throws -> String) {
        guard !busy else { return }
        busy = true
        Task {
            defer { busy = false }
            do {
                guard let id = devicePath(model) else { throw AdminError.unavailable }
                note = try await action(model.mobileClient(), id)
                await refresh()
                if let pendingCreatedID, members.contains(where: { $0.id == pendingCreatedID }) {
                    navigationPath.append(pendingCreatedID)
                    self.pendingCreatedID = nil
                }
            } catch { note = error.localizedDescription; await refresh() }
        }
    }

    private func refresh() async {
        guard let id = devicePath(model), model.pairing?.adminURL != nil else { return }
        do {
            let client = try model.mobileClient()
            let list = try await client.json("GET", "/api/identity/members")
            let settings = try await client.json("GET", "/api/identity/devices/\(id)/settings")
            if let minimum = list["min_samples_per_modality"] as? Int, (1...10).contains(minimum) {
                requiredSamples = minimum
            }
            members = ((list["members"] as? [[String: Any]]) ?? []).compactMap(FamilyMember.init)
            recognition = settings["enabled"] as? Bool ?? false
        } catch { note = error.localizedDescription }
    }
}

private struct MemberDetailView: View {
    @EnvironmentObject private var model: ParentModel
    let member: FamilyMember
    let requiredSamples: Int
    let onChanged: () async -> Void
    @State private var samples: [[String: Any]] = []
    @State private var voiceCount = 0
    @State private var faceCount = 0
    @State private var ticketID: String?
    @State private var ticketState = ""
    @State private var note = ""
    @State private var busy = false
    @State private var deleteSampleID: String?
    @State private var deleteMember = false
    @State private var phoneRecorder: PhoneVoiceRecorder?
    @State private var phoneRecording = false
    @State private var voiceWorking = false
    @State private var voiceStatus = ""
    @State private var showPhoneCamera = false
    @State private var faceStatus = ""

    var body: some View {
        Form {
            Section(member.name) {
                LabeledContent("声纹", value: "\(voiceCount) 段")
                LabeledContent("人脸", value: "\(faceCount) 张")
                Text(member.readinessText(requiredSamples: requiredSamples))
                    .font(.subheadline)
                    .foregroundStyle(member.ready ? Color.green : Color.orange)
                Text("每种至少录入 \(requiredSamples) 份不同样本；可直接用手机麦克风和摄像头录入。原始素材不传到 iCloud。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("用手机录入") {
                Button {
                    Task { await togglePhoneRecording() }
                } label: {
                    HStack {
                        Text(phoneRecording ? "结束录音并保存声纹" : "用手机麦克风录入声纹")
                        if voiceWorking { Spacer(); ProgressView() }
                    }
                }
                .disabled(busy)
                .accessibilityIdentifier("phoneVoiceEnrollmentButton")
                Text(phoneRecording ? "正在录音。请本人对着手机说话，3 秒后点“结束录音并保存声纹”。" :
                     "本人对着手机说 3–20 秒，录满 20 秒会自动停下，再点保存。只送到家中 Mac 提取特征；下一份请换一句话。")
                    .font(.caption).foregroundStyle(.secondary)
                if !voiceStatus.isEmpty {
                    Text(voiceStatus).font(.footnote).foregroundStyle(.secondary)
                        .accessibilityIdentifier("voiceEnrollmentStatus")
                        .task(id: phoneRecording) {
                            guard phoneRecording else { return }
                            for second in 1...20 {
                                try? await Task.sleep(for: .seconds(1))
                                guard !Task.isCancelled, phoneRecording else { return }
                                let lowInput = (phoneRecorder?.inputLevel() ?? -160) < -43
                                voiceStatus = second == 20 ? "已录满 20 秒，请点「结束录音并保存声纹」" :
                                    (lowInput ? "录音中 \(second) 秒 · 收音偏低，请靠近手机底部麦克风" :
                                     "录音中 \(second) 秒 · 已收到声音，至少录满 3 秒后点结束")
                            }
                        }
                }
                Button("用手机拍照录入人脸") {
                    guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
                        faceStatus = "此设备没有可用摄像头"
                        return
                    }
                    faceStatus = "请让本人正对摄像头拍照，再点相机中的「使用照片」完成上传"
                    showPhoneCamera = true
                }
                .disabled(busy || phoneRecording)
                .accessibilityIdentifier("phoneFaceEnrollmentButton")
                Text("拍照后还需点「使用照片」；返回此页看到「人脸已保存」和数量增加才算成功。照片不会存入相册或 iCloud。")
                    .font(.caption).foregroundStyle(.secondary)
                if !faceStatus.isEmpty {
                    Text(faceStatus).font(.footnote).foregroundStyle(.secondary)
                        .accessibilityIdentifier("faceEnrollmentStatus")
                }
            }
            Section {
                DisclosureGroup("备用：用小陆录入") {
                    Button("开启 60 秒声纹录入") { perform { client, id in
                        let result = try await client.json("POST", "/api/identity/voice-tickets",
                                                           body: ["device_id": id, "person_id": member.id])
                        ticketID = result["ticket_id"] as? String
                        ticketState = "请让 \(member.name) 先说“小陆同学”，看到“聆听”后独自说话约 3–12 秒"
                        return "声纹窗口已开启"
                    } }.disabled(busy)
                        .accessibilityIdentifier("startVoiceEnrollmentButton")
                    if let ticketID {
                        Text(ticketState).font(.subheadline)
                        Button("取消本次录入") { perform { client, _ in
                            _ = try await client.json("DELETE", "/api/identity/voice-tickets/\(ticketID)")
                            self.ticketID = nil
                            return "已取消本次录入"
                        } }
                        .task(id: ticketID) { await pollTicket(ticketID) }
                    }
                    Button("让小陆拍摄一张人脸") { perform { client, id in
                        _ = try await client.json("POST", "/api/identity/members/\(member.id)/capture-face",
                                                  body: ["device_id": id])
                        return "拍摄请求已完成；请换一个角度再录一张"
                    } }.disabled(busy)
                }
            }
            Section("已保存样本") {
                if samples.isEmpty { Text("暂无样本").foregroundStyle(.secondary) }
                ForEach(samples.indices, id: \.self) { index in
                    let sample = samples[index]
                    let id = sample["sample_id"] as? String ?? ""
                    HStack {
                        Text(sample["modality"] as? String == "face" ? "人脸" : "声纹")
                        Spacer()
                        Button("删除") { deleteSampleID = id }.tint(.red).disabled(id.isEmpty)
                    }
                }
            }
            Section {
                Button("删除这位家人及全部身份样本", role: .destructive) { deleteMember = true }
            }
            if !note.isEmpty { Section { Text(note).font(.footnote).foregroundStyle(.secondary) } }
        }
        .navigationTitle(member.name)
        .sheet(isPresented: $showPhoneCamera) {
            PhoneCameraCapture { image in
                showPhoneCamera = false
                if let image { Task { await uploadPhoneFace(image) } }
                else { faceStatus = "已取消拍照" }
            }
            .ignoresSafeArea()
        }
        .onDisappear { phoneRecorder?.cancel(); phoneRecording = false }
        .task { await refresh() }
        .refreshable { await refresh() }
        .alert("删除样本？", isPresented: Binding(get: { deleteSampleID != nil }, set: { if !$0 { deleteSampleID = nil } })) {
            Button("删除", role: .destructive) { if let id = deleteSampleID { perform { client, _ in
                _ = try await client.json("DELETE", "/api/identity/members/\(member.id)/samples/\(id)")
                deleteSampleID = nil
                return "样本已删除"
            } } }
            Button("取消", role: .cancel) { deleteSampleID = nil }
        } message: { Text("删除后该样本不能恢复。") }
        .alert("删除家人？", isPresented: $deleteMember) {
            Button("删除全部身份样本", role: .destructive) { perform { client, _ in
                _ = try await client.json("DELETE", "/api/identity/members/\(member.id)")
                return "家人身份已撤销"
            } }
            Button("取消", role: .cancel) { }
        } message: { Text("这会撤销此人的声纹与人脸样本。") }
    }

    private func perform(_ action: @escaping (MobileAdminClient, String) async throws -> String) {
        guard !busy else { return }
        busy = true
        Task {
            defer { busy = false }
            do {
                guard let id = devicePath(model) else { throw AdminError.unavailable }
                note = try await action(model.mobileClient(), id)
                await refresh(); await onChanged()
            } catch { note = error.localizedDescription }
        }
    }

    private func togglePhoneRecording() async {
        guard !busy else { return }
        busy = true
        voiceWorking = true
        defer { busy = false; voiceWorking = false }
        do {
            if phoneRecording {
                voiceStatus = "正在保存声纹…"
                guard let phoneRecorder else { throw AdminError.invalidInput("录音尚未开始") }
                let wav = try phoneRecorder.finish()
                self.phoneRecorder = nil
                phoneRecording = false
                note = "正在将录音送到家中 Mac 提取声纹…"
                _ = try await model.mobileClient().uploadVoice(wav, for: member.id)
                note = "声纹已保存；请查看顶部的识别准备状态"
                voiceStatus = note
                await refresh(); await onChanged()
            } else {
                voiceStatus = "正在请求麦克风权限…"
                let recorder = PhoneVoiceRecorder()
                try await recorder.start()
                phoneRecorder = recorder
                phoneRecording = true
                note = "正在录音，3–20 秒后点结束"
                voiceStatus = note
            }
        } catch {
            phoneRecorder?.cancel()
            phoneRecorder = nil
            phoneRecording = false
            note = error.localizedDescription
            voiceStatus = note
        }
    }

    private func uploadPhoneFace(_ image: UIImage) async {
        guard !busy else { return }
        busy = true
        faceStatus = "正在处理照片并送到家中 Mac…"
        defer { busy = false }
        do {
            let jpeg = try faceJPEG(image)
            _ = try await model.mobileClient().uploadFace(jpeg, for: member.id)
            faceStatus = "人脸已保存；请查看顶部的识别准备状态"
            await refresh(); await onChanged()
        } catch {
            faceStatus = error.localizedDescription
        }
    }

    private func refresh() async {
        do {
            samples = try await model.mobileClient().jsonArray("GET", "/api/identity/members/\(member.id)/samples")
            voiceCount = samples.filter { $0["modality"] as? String == "voice" }.count
            faceCount = samples.filter { $0["modality"] as? String == "face" }.count
        } catch { note = error.localizedDescription }
    }

    private func pollTicket(_ id: String) async {
        for _ in 0..<22 {
            guard !Task.isCancelled, ticketID == id else { return }
            do {
                let result = try await model.mobileClient().json("GET", "/api/identity/voice-tickets/\(id)")
                let state = result["state"] as? String ?? "waiting"
                if state != "waiting" {
                    ticketState = state == "completed" ? "声纹已保存；再换一句话录入一次" : "本次录入\(state)"
                    ticketID = nil
                    await refresh(); await onChanged()
                    return
                }
            } catch { ticketState = error.localizedDescription; return }
            try? await Task.sleep(for: .seconds(3))
        }
        ticketState = "录入窗口结束；请刷新查看样本"
        ticketID = nil
    }
}

struct ManagementHomeView: View {
    @EnvironmentObject private var model: ParentModel
    var body: some View {
        NavigationStack {
            List {
                Section {
                    Text("这些设置需手机与家中 Mac 连接同一 Wi-Fi。外出时可在「小狗」查看状态、拍照和请求停止。")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                Section("学习与记忆") {
                    NavigationLink { TextChatView().environmentObject(model) } label: { Label("文字陪伴", systemImage: "bubble.left.and.bubble.right") }
                    NavigationLink { LearningSettingsView().environmentObject(model) } label: { Label("英语练习设置", systemImage: "character.book.closed") }
                    NavigationLink { MemoryManagementView().environmentObject(model) } label: { Label("个人记忆与故事", systemImage: "brain.head.profile") }
                    NavigationLink { EnglishCheckView().environmentObject(model) } label: { Label("英语观察记录", systemImage: "checkmark.seal") }
                    NavigationLink { RoutineManagementView().environmentObject(model) } label: { Label("生活小任务", systemImage: "list.bullet.clipboard") }
                }
                Section("设备与环境") {
                    NavigationLink { DeviceManagementView().environmentObject(model) } label: { Label("设备设置", systemImage: "gearshape") }
                    NavigationLink { PatrolManagementView().environmentObject(model) } label: { Label("定时原地巡看", systemImage: "camera.viewfinder") }
                    NavigationLink { PhotoGalleryView().environmentObject(model) } label: { Label("最近照片", systemImage: "photo.on.rectangle") }
                    NavigationLink { ActionManagementView().environmentObject(model) } label: { Label("动作与离线动作库", systemImage: "figure.walk") }
                    NavigationLink { GestureManagementView().environmentObject(model) } label: { Label("手势控制", systemImage: "hand.raised") }
                    NavigationLink { GatewayManagementView().environmentObject(model) } label: { Label("连接与维护", systemImage: "network") }
                }
                if model.pairing?.adminURL == nil {
                    Section { Text("请在「连接」导入带手机管理连接的配置，才能使用这些功能。")
                        .foregroundStyle(.secondary) }
                }
            }
            .navigationTitle("管理小陆")
        }
    }
}

private struct LearningSettingsView: View {
    @EnvironmentObject private var model: ParentModel
    @State private var person = ""
    @State private var minutes = 5
    @State private var theme = "animals"
    @State private var saveProgress = true
    @State private var photoLearning = false
    @State private var used = ""
    @State private var note = ""
    @State private var busy = false
    private let themes = [("animals", "动物"), ("colors", "颜色"), ("greetings", "问候"),
                          ("objects", "物品"), ("body_parts", "身体")]

    var body: some View {
        Form {
            Section("对象") {
                TextField("留空为访客；或输入成员 ID", text: $person)
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                    .onSubmit { Task { await load() } }
                Button("读取此人的设置和今日用时") { Task { await load() } }.disabled(busy)
            }
            Section("每日英语练习") {
                Stepper("每天 \(minutes) 分钟", value: $minutes, in: 1...20)
                Picker("主题", selection: $theme) {
                    ForEach(themes, id: \.0) { item in Text(item.1).tag(item.0) }
                }
                Toggle("保存英语进度", isOn: $saveProgress)
                Toggle("允许用 Puppy 拍照学英语", isOn: $photoLearning)
                if !used.isEmpty { Text(used).font(.footnote).foregroundStyle(.secondary) }
                Button("保存练习安排") { Task { await save() } }.disabled(busy)
                Button("结束本次英语计时") { Task { await stop() } }.disabled(busy)
            }
            if !note.isEmpty { Section { Text(note).font(.footnote).foregroundStyle(.secondary) } }
        }
        .navigationTitle("英语练习")
        .task { await load() }
    }

    private func path() throws -> String {
        guard let id = devicePath(model), person.isEmpty || safeMember(person) else {
            throw AdminError.invalidInput("请输入有效成员 ID")
        }
        return "/api/learning/\(id)" + (person.isEmpty ? "" : "/members/\(person)")
    }

    private func show(_ result: [String: Any]) {
        if let settings = result["settings"] as? [String: Any] {
            minutes = settings["daily_minutes"] as? Int ?? minutes
            theme = settings["theme"] as? String ?? theme
            saveProgress = settings["save_progress"] as? Bool ?? saveProgress
            photoLearning = settings["photo_learning"] as? Bool ?? photoLearning
        }
        if let remaining = result["remaining_seconds"] as? Double {
            used = "今天还可练习约 \(Int(remaining / 60)) 分钟"
        }
    }

    private func load() async {
        guard !busy else { return }; busy = true; defer { busy = false }
        do { show(try await model.mobileClient().json("GET", path())); note = "已从家中 Mac 读取" }
        catch { note = error.localizedDescription }
    }
    private func save() async {
        guard !busy else { return }; busy = true; defer { busy = false }
        do {
            let result = try await model.mobileClient().json("PATCH", path(), body: [
                "daily_minutes": minutes, "theme": theme, "save_progress": saveProgress, "photo_learning": photoLearning])
            show(result); note = "英语安排已保存"
        } catch { note = error.localizedDescription }
    }
    private func stop() async {
        guard !busy else { return }; busy = true; defer { busy = false }
        do { show(try await model.mobileClient().json("POST", path() + "/stop")); note = "本次计时已结束" }
        catch { note = error.localizedDescription }
    }
}

private struct MemoryManagementView: View {
    @EnvironmentObject private var model: ParentModel
    @State private var person = ""
    @State private var entries: [[String: Any]] = []
    @State private var story = ""
    @State private var note = ""
    @State private var deleteKey: String?
    @State private var clearAll = false

    var body: some View {
        Form {
            Section("选择家人") {
                TextField("成员 ID", text: $person).textInputAutocapitalization(.never).autocorrectionDisabled()
                Button("读取记忆") { Task { await load() } }.disabled(!safeMember(person))
            }
            Section("个人记忆") {
                if entries.isEmpty { Text("没有已保存的记忆").foregroundStyle(.secondary) }
                ForEach(entries.indices, id: \.self) { index in
                    let row = entries[index]
                    let key = row["memory_key"] as? String ?? ""
                    VStack(alignment: .leading, spacing: 5) {
                        Text(row["value"] as? String ?? "").font(.body)
                        HStack {
                            Text(key).font(.caption.monospaced()).foregroundStyle(.secondary)
                            Spacer()
                            Button("删除", role: .destructive) { deleteKey = key }.disabled(key.isEmpty)
                        }
                    }
                }
            }
            if !story.isEmpty { Section("故事进度") { Text(story) } }
            Section {
                Button("删除故事进度", role: .destructive) { Task { await removeStory() } }
                Button("删除此人的全部记忆", role: .destructive) { clearAll = true }
            }
            if !note.isEmpty { Section { Text(note).font(.footnote).foregroundStyle(.secondary) } }
        }
        .navigationTitle("记忆与故事")
        .alert("删除这条记忆？", isPresented: Binding(get: { deleteKey != nil }, set: { if !$0 { deleteKey = nil } })) {
            Button("删除", role: .destructive) { Task { await removeKey() } }
            Button("取消", role: .cancel) { deleteKey = nil }
        } message: { Text("删除后无法恢复。") }
        .alert("删除此人的全部记忆？", isPresented: $clearAll) {
            Button("全部删除", role: .destructive) { Task { await removeAll() } }
            Button("取消", role: .cancel) { }
        } message: { Text("也会清除这个人的英语进度和故事记录。") }
    }

    private func path() throws -> String {
        guard let id = devicePath(model), safeMember(person) else { throw AdminError.invalidInput("请先填写有效成员 ID") }
        return "/api/memory/\(id)/\(person)"
    }
    private func load() async {
        do {
            let result = try await model.mobileClient().json("GET", path())
            entries = result["memories"] as? [[String: Any]] ?? []
            if let text = result["story"] as? String { story = text }
            else if let value = result["story"], !(value is NSNull),
                    let data = try? JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys]) {
                story = String(decoding: data, as: UTF8.self)
            } else { story = "" }
            note = "已读取家中 Mac 保存的内容"
        } catch { note = error.localizedDescription }
    }
    private func removeKey() async {
        guard let key = deleteKey,
              key.range(of: "^[A-Za-z0-9_:.-]{1,128}$", options: .regularExpression) != nil else {
            note = "记忆键无效"; return
        }
        do { _ = try await model.mobileClient().json("DELETE", path() + "/" + key); deleteKey = nil; await load() }
        catch { note = error.localizedDescription }
    }
    private func removeStory() async {
        do {
            guard let id = devicePath(model), safeMember(person) else { throw AdminError.invalidInput("请先填写成员 ID") }
            _ = try await model.mobileClient().json("DELETE", "/api/story/\(id)/\(person)"); await load()
        } catch { note = error.localizedDescription }
    }
    private func removeAll() async {
        do { _ = try await model.mobileClient().json("DELETE", path()); await load() }
        catch { note = error.localizedDescription }
    }
}

private struct DeviceManagementView: View {
    @EnvironmentObject private var model: ParentModel
    @State private var volume = 20
    @State private var mode = "chat"
    @State private var style = "gentle"
    @State private var greeting = "none"
    @State private var profileRevision = 0
    @State private var online = false
    @State private var autoSleepEnabled = false
    @State private var clockTrusted = false
    @State private var note = ""
    @State private var busy = false

    var body: some View {
        Form {
            Section("小陆状态") {
                Label(online ? "设备在线" : "设备未确认在线", systemImage: online ? "checkmark.circle" : "wifi.slash")
                Button("刷新实际状态") { Task { await load() } }.disabled(busy)
            }
            Section("声音") {
                Stepper("音量 \(volume) / 60", value: $volume, in: 0...60)
                HStack {
                    Button("保存音量") { Task { await call("POST", "/api/devices/{id}/volume", ["volume": volume], "已向设备发送音量设置") } }
                    Button("立即停止说话", role: .destructive) { Task { await call("POST", "/api/devices/{id}/audio/stop", nil, "已请求停止发声") } }
                }.disabled(busy || !online)
            }
            Section("休眠与背灯") {
                Label(autoSleepEnabled ? "自动休眠已开启" : "自动休眠未开启",
                      systemImage: autoSleepEnabled ? "moon.zzz.fill" : "moon.zzz")
                Text("每天 21:00–07:00 进入深度休眠；白天连续 30 分钟没有互动也会休眠。按顶部键唤醒。")
                    .font(.footnote).foregroundStyle(.secondary)
                if !clockTrusted { Text("尚未完成设备校时，自动休眠暂不可用。")
                    .font(.footnote).foregroundStyle(.orange) }
                Button(autoSleepEnabled ? "关闭自动休眠" : "开启自动休眠") {
                    Task { await call("POST", "/api/devices/{id}/sleep-policy",
                                      ["enabled": !autoSleepEnabled],
                                      autoSleepEnabled ? "已关闭自动休眠" : "已开启自动休眠") }
                }.disabled(busy || !online || (!autoSleepEnabled && !clockTrusted))
                Button("测试背灯（亮 1 秒）") {
                    Task { await call("POST", "/api/devices/{id}/lamp-test", nil, "已发送背灯测试") }
                }.disabled(busy || !online)
            }
            Section("对话") {
                Picker("模式", selection: $mode) {
                    Text("自由问答").tag("chat"); Text("英语").tag("english"); Text("故事").tag("story")
                }
                Button("保存对话模式") { Task { await call("PUT", "/api/devices/{id}/mode", ["mode": mode], "对话模式已保存") } }
                    .disabled(busy)
            }
            Section("小陆的性格") {
                Picker("性格", selection: $style) {
                    Text("温和").tag("gentle"); Text("活泼").tag("lively"); Text("耐心").tag("patient")
                }
                Picker("见面动作", selection: $greeting) {
                    Text("不做动作").tag("none"); Text("挥手").tag("wave"); Text("坐下").tag("sit")
                    Text("伸懒腰").tag("stretch"); Text("拥抱").tag("hug")
                }
                Button("保存性格") { Task { await saveProfile() } }.disabled(busy)
            }
            if !note.isEmpty { Section { Text(note).font(.footnote).foregroundStyle(.secondary) } }
        }
        .navigationTitle("设备设置")
        .task { await load() }
    }

    private func call(_ method: String, _ template: String, _ body: [String: Any]?, _ success: String) async {
        guard !busy else { return }; busy = true; defer { busy = false }
        do {
            guard let id = devicePath(model) else { throw AdminError.unavailable }
            let path = template.replacingOccurrences(of: "{id}", with: id)
            let result = try await model.mobileClient().json(method, path, body: body)
            if result["ok"] as? Bool == false {
                throw AdminError.invalidInput(result["error"] as? String ?? "设备未执行")
            }
            if method != "POST" || template.contains("volume") || template.contains("sleep-policy") {
                await load()
            }
            note = success
        } catch { note = error.localizedDescription }
    }
    private func load() async {
        do {
            guard let id = devicePath(model) else { throw AdminError.unavailable }
            let client = try model.mobileClient()
            let status = try await client.json("GET", "/api/status")
            let devices = status["devices"] as? [String: [String: Any]] ?? [:]
            online = devices[id]?["online"] as? Bool ?? (devices[id] != nil)
            let speaker = devices[id]?["audio_speaker"] as? [String: Any]
            volume = speaker?["volume"] as? Int ?? volume
            autoSleepEnabled = devices[id]?["auto_sleep_enabled"] as? Bool ?? autoSleepEnabled
            clockTrusted = devices[id]?["clock_trusted"] as? Bool ?? clockTrusted
            let pet = try await client.json("GET", "/api/pet/\(id)")
            mode = pet["conversation_mode"] as? String ?? mode
            let profile = try await client.json("GET", "/api/pet-profile/\(id)")
            style = profile["style"] as? String ?? style
            greeting = profile["greeting_action"] as? String ?? greeting
            profileRevision = profile["revision"] as? Int ?? profileRevision
            note = "已读取网关与设备状态"
        } catch { note = error.localizedDescription }
    }
    private func saveProfile() async {
        guard !busy else { return }; busy = true; defer { busy = false }
        do {
            guard let id = devicePath(model) else { throw AdminError.unavailable }
            let result = try await model.mobileClient().json("PUT", "/api/pet-profile/\(id)", body: [
                "style": style, "greeting_action": greeting, "expected_revision": profileRevision])
            profileRevision = result["revision"] as? Int ?? profileRevision
            note = "性格已保存"
        } catch { note = error.localizedDescription }
    }
}

private struct PatrolManagementView: View {
    @EnvironmentObject private var model: ParentModel
    @State private var enabled = false
    @State private var interval = 60
    @State private var start = "09:00"
    @State private var end = "21:00"
    @State private var describe = false
    @State private var retention = 7
    @State private var runs: [[String: Any]] = []
    @State private var selectedImage: UIImage?
    @State private var note = ""
    @State private var busy = false

    var body: some View {
        Form {
            Section("原地巡看计划") {
                Toggle("启用定时拍照", isOn: $enabled)
                Stepper("每 \(interval) 分钟", value: $interval, in: 15...1440, step: 15)
                TextField("开始时间 HH:mm", text: $start)
                TextField("结束时间 HH:mm", text: $end)
                Toggle("用本地模型描述照片", isOn: $describe)
                Stepper("本地照片保留 \(retention) 天", value: $retention, in: 1...30)
                Text("小陆固定位置拍照，不移动、不出声。离线或错过的时段不会补拍。")
                    .font(.caption).foregroundStyle(.secondary)
                Button("保存计划") { Task { await save() } }.disabled(busy)
            }
            Section("最近巡看") {
                if runs.isEmpty { Text("暂无巡看记录").foregroundStyle(.secondary) }
                ForEach(runs.indices, id: \.self) { index in
                    let run = runs[index]
                    VStack(alignment: .leading, spacing: 4) {
                        Text(run["state"] as? String ?? "未知状态")
                        if let description = run["description"] as? String { Text(description).font(.footnote) }
                        if let time = run["scheduled_at"] as? Double {
                            Text(Date(timeIntervalSince1970: time).formatted()).font(.caption).foregroundStyle(.secondary)
                        }
                        if let photoID = run["photo_id"] as? String, UUID(uuidString: photoID) != nil {
                            Button("查看本次照片") { Task { await showPhoto(photoID) } }
                        }
                    }
                }
                Button("刷新计划和记录") { Task { await load() } }.disabled(busy)
                Button("清理到期的本地巡看照片") { Task { await cleanup() } }.disabled(busy)
            }
            if let selectedImage {
                Section("单次巡看照片") {
                    Image(uiImage: selectedImage).resizable().scaledToFit()
                    Text("照片只覆盖小陆当前镜头视野。")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            if !note.isEmpty { Section { Text(note).font(.footnote).foregroundStyle(.secondary) } }
        }
        .navigationTitle("定时巡看")
        .task { await load() }
    }
    private func load() async {
        do {
            guard let id = devicePath(model) else { throw AdminError.unavailable }
            let client = try model.mobileClient()
            let config = try await client.json("GET", "/api/patrol/devices/\(id)")
            enabled = config["enabled"] as? Bool ?? false
            interval = config["interval_minutes"] as? Int ?? 60
            start = config["window_start"] as? String ?? "09:00"
            end = config["window_end"] as? String ?? "21:00"
            describe = config["describe"] as? Bool ?? false
            retention = config["retention_days"] as? Int ?? 7
            let history = try await client.json("GET", "/api/patrol/devices/\(id)/runs")
            runs = history["runs"] as? [[String: Any]] ?? []
            note = enabled ? "计划已启用" : "计划已关闭"
        } catch { note = error.localizedDescription }
    }
    private func save() async {
        guard !busy else { return }; busy = true; defer { busy = false }
        do {
            guard let id = devicePath(model), start.range(of: "^[0-2][0-9]:[0-5][0-9]$", options: .regularExpression) != nil,
                  end.range(of: "^[0-2][0-9]:[0-5][0-9]$", options: .regularExpression) != nil else {
                throw AdminError.invalidInput("请使用 HH:mm 时间格式")
            }
            _ = try await model.mobileClient().json("PUT", "/api/patrol/devices/\(id)", body: [
                "enabled": enabled, "interval_minutes": interval, "window_start": start,
                "window_end": end, "timezone": "Asia/Shanghai", "describe": describe, "retention_days": retention])
            note = "巡看计划已保存；不会立即拍照"
        } catch { note = error.localizedDescription }
    }
    private func showPhoto(_ id: String) async {
        do {
            let data = try await model.mobileClient().request("GET", "/api/patrol/photos/\(id)")
            guard let image = UIImage(data: data) else { throw AdminError.invalidResponse }
            selectedImage = image; note = "已读取这次巡看的本地照片"
        } catch { note = error.localizedDescription }
    }
    private func cleanup() async {
        do {
            let result = try await model.mobileClient().json("POST", "/api/patrol/cleanup")
            await load()
            note = "已清理 \(result["deleted"] as? Int ?? 0) 张到期本地照片；iCloud 副本未更改"
        } catch { note = error.localizedDescription }
    }
}

private struct ActionManagementView: View {
    @EnvironmentObject private var model: ParentModel
    @State private var actions: [String: [[String: Any]]] = [:]
    @State private var drafts: [[String: Any]] = []
    @State private var name = ""
    @State private var editedSteps = [EditedActionStep()]
    @State private var offlineNames: Set<String> = []
    @State private var offlineEligible: Set<String> = []
    @State private var offlineVersion = 0
    @State private var offlineState = ""
    @State private var note = ""
    @State private var busy = false
    @State private var runName: String?
    @State private var deleteName: String?

    var body: some View {
        Form {
            Section("自定义动作") {
                TextField("动作名称", text: $name)
                ForEach($editedSteps) { $step in
                    VStack(alignment: .leading, spacing: 5) {
                        Picker("动作", selection: $step.action) {
                            Text("挥手").tag("wave"); Text("坐下").tag("sit")
                            Text("伸懒腰").tag("stretch"); Text("拥抱").tag("hug")
                        }
                        Stepper("完成后等待 \(step.waitMS) 毫秒", value: $step.waitMS, in: 0...3000, step: 100)
                        if editedSteps.count > 1 {
                            Button("删除这一步", role: .destructive) {
                                editedSteps.removeAll { $0.id == step.id }
                            }
                        }
                    }
                }
                Button("添加一步") { if editedSteps.count < 12 { editedSteps.append(EditedActionStep()) } }
                    .disabled(editedSteps.count >= 12)
                Button("保存这一动作") { Task { await saveAction() } }.disabled(busy || name.isEmpty)
                Text("最多 12 步，只支持四种短动作，总等待不超过 20 秒。保存不会让小陆运动。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("已保存动作") {
                if actions.isEmpty { Text("暂无动作").foregroundStyle(.secondary) }
                ForEach(actions.keys.sorted(), id: \.self) { action in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(action).font(.headline)
                        Text(stepsText(actions[action] ?? [])).font(.caption).foregroundStyle(.secondary)
                        HStack {
                            Button("编辑") {
                                name = action
                                editedSteps = (actions[action] ?? []).map { step in
                                    EditedActionStep(action: step["action"] as? String ?? "wave",
                                                     waitMS: step["wait_ms"] as? Int ?? 0)
                                }
                                if editedSteps.isEmpty { editedSteps = [EditedActionStep()] }
                            }
                            Button("执行") { runName = action }.disabled(busy)
                            Button("删除", role: .destructive) { deleteName = action }.disabled(busy)
                        }
                    }
                }
            }
            Section("机身离线动作库") {
                Text("同步最多 3 个已保存动作；只保存到小陆，不会自动执行。")
                    .font(.caption).foregroundStyle(.secondary)
                ForEach(actions.keys.sorted(), id: \.self) { action in
                    Toggle(action, isOn: Binding(get: { offlineNames.contains(action) }, set: { selected in
                        if selected && offlineNames.count < 3 { offlineNames.insert(action) }
                        else if !selected { offlineNames.remove(action) }
                    })).disabled(!offlineEligible.contains(action))
                }
                if !offlineState.isEmpty { Text(offlineState).font(.footnote) }
                Button("同步所选动作到小陆") { Task { await syncOffline() } }.disabled(busy)
                Button("重试同一版本和快照") { Task { await retryOffline() } }
                    .disabled(busy || offlineVersion == 0)
                Button("清空机身离线动作库") { offlineNames = []; Task { await syncOffline() } }
                    .disabled(busy)
                Button("刷新离线库状态") { Task { await load() } }.disabled(busy)
            }
            Section("孩子提议的新把戏") {
                if drafts.isEmpty { Text("暂无待审核草稿").foregroundStyle(.secondary) }
                ForEach(drafts.indices, id: \.self) { index in
                    let draft = drafts[index]
                    let id = draft["draft_id"] as? String ?? ""
                    let state = draft["state"] as? String ?? ""
                    VStack(alignment: .leading, spacing: 5) {
                        Text(draft["name"] as? String ?? "未命名").font(.headline)
                        Text(draft["source_text"] as? String ?? "").font(.caption)
                        Text(stepsText(draft["steps"] as? [[String: Any]] ?? []))
                            .font(.caption).foregroundStyle(.secondary)
                        Text("状态：\(state) · 版本 \(draft["version"] as? Int ?? 0)")
                            .font(.caption2).foregroundStyle(.secondary)
                        if state == "draft", !id.isEmpty {
                            HStack {
                                NavigationLink("编辑步骤") {
                                    DraftReviewView(draftID: id, onChanged: { await load() })
                                        .environmentObject(model)
                                }
                                Button("批准保存") { Task { await review(id, draft, approve: true) } }
                                Button("拒绝", role: .destructive) { Task { await review(id, draft, approve: false) } }
                            }.disabled(busy)
                        } else if state == "approved", draft["can_revoke"] as? Bool == true, !id.isEmpty {
                            Button("撤销对应已批准动作", role: .destructive) {
                                Task { await revokeDraft(id, draft) }
                            }.disabled(busy)
                        }
                    }
                }
                Button("刷新草稿") { Task { await load() } }.disabled(busy)
            }
            if !note.isEmpty { Section { Text(note).font(.footnote).foregroundStyle(.secondary) } }
        }
        .navigationTitle("动作管理")
        .task { await load() }
        .alert("执行这个动作？", isPresented: Binding(get: { runName != nil }, set: { if !$0 { runName = nil } })) {
            Button("执行") { if let action = runName { Task { await run(action) } } }
            Button("取消", role: .cancel) { runName = nil }
        } message: { Text("请确认小陆摆放平稳、周围有空间，现场可长按停止。") }
        .alert("删除这个动作？", isPresented: Binding(get: { deleteName != nil }, set: { if !$0 { deleteName = nil } })) {
            Button("删除", role: .destructive) { if let action = deleteName { Task { await removeAction(action) } } }
            Button("取消", role: .cancel) { deleteName = nil }
        } message: { Text("已同步的离线副本需要另行更新。") }
    }

    private func stepsText(_ steps: [[String: Any]]) -> String {
        steps.map { "\($0["action"] as? String ?? "等待") \($0["wait_ms"] as? Int ?? 0) ms" }.joined(separator: " → ")
    }
    private func load() async {
        do {
            guard let id = devicePath(model) else { throw AdminError.unavailable }
            let client = try model.mobileClient()
            let all = try await client.json("GET", "/api/actions")
            actions = all as? [String: [[String: Any]]] ?? [:]
            let library = try await client.json("GET", "/api/offline-library/devices/\(id)")
            let latest = library["latest"] as? [String: Any]
            offlineVersion = latest?["version"] as? Int ?? 0
            offlineEligible = Set(((library["catalog"] as? [[String: Any]]) ?? [])
                .filter { $0["eligible"] as? Bool == true }.compactMap { $0["name"] as? String })
            offlineNames = offlineNames.intersection(offlineEligible)
            offlineState = latest.map { "最近版本 \($0["version"] as? Int ?? 0) · \($0["state"] as? String ?? "未知")" } ?? "尚未同步"
            let proposal = try await client.json("GET", "/api/action-drafts/devices/\(id)")
            drafts = proposal["drafts"] as? [[String: Any]] ?? []
            note = "已读取家中动作库"
        } catch { note = error.localizedDescription }
    }
    private func saveAction() async {
        guard !busy else { return }; busy = true; defer { busy = false }
        do {
            guard name.range(of: "^[A-Za-z0-9_\\u4e00-\\u9fff-]{1,32}$", options: .regularExpression) != nil else {
                throw AdminError.invalidInput("名称须为 1–32 个字母、数字或中文字符")
            }
            guard (1...12).contains(editedSteps.count), editedSteps.reduce(0, { $0 + $1.waitMS }) <= 20000 else {
                throw AdminError.invalidInput("动作最多 12 步，总等待不能超过 20 秒")
            }
            let steps = editedSteps.map { ["action": $0.action, "wait_ms": $0.waitMS] as [String: Any] }
            _ = try await model.mobileClient().json("PUT", "/api/actions", body: ["name": name, "steps": steps])
            note = "动作已保存；小陆没有开始运动"
            await load()
        } catch { note = error.localizedDescription }
    }
    private func run(_ action: String) async {
        runName = nil
        guard !busy else { return }; busy = true; defer { busy = false }
        do {
            guard let id = devicePath(model), action.range(of: "^[A-Za-z0-9_\\u4e00-\\u9fff-]{1,32}$", options: .regularExpression) != nil else {
                throw AdminError.invalidInput("动作名称无效")
            }
            let receipt = try await model.mobileClient().json("POST", "/api/devices/\(id)/action/\(action)")
            note = receipt["accepted"] as? Bool == true
                ? "家中 Mac 已接收动作，正在等待小陆完成"
                : "动作提交未确认，请先查看设备状态，不要重复点击"
        } catch { note = error.localizedDescription }
    }
    private func removeAction(_ action: String) async {
        deleteName = nil
        guard !busy else { return }; busy = true; defer { busy = false }
        do {
            guard let steps = actions[action] else { throw AdminError.invalidInput("动作已变化，请刷新") }
            _ = try await model.mobileClient().json("DELETE", "/api/actions/\(action)", body: ["expected_steps": steps])
            await load(); note = "动作已删除"
        } catch { note = error.localizedDescription }
    }
    private func syncOffline() async {
        guard !busy else { return }; busy = true; defer { busy = false }
        do {
            guard let id = devicePath(model) else { throw AdminError.unavailable }
            let result = try await model.mobileClient().json("POST", "/api/offline-library/devices/\(id)/sync",
                                                           body: ["names": offlineNames.sorted()])
            let latest = result["latest"] as? [String: Any]
            offlineState = "同步版本 \(latest?["version"] as? Int ?? 0) · \(latest?["state"] as? String ?? "待核对")"
            note = "已请求保存到小陆；不会自动运动"
        } catch { note = error.localizedDescription }
    }
    private func retryOffline() async {
        guard !busy else { return }; busy = true; defer { busy = false }
        do {
            guard let id = devicePath(model), offlineVersion > 0 else { throw AdminError.unavailable }
            _ = try await model.mobileClient().json("POST", "/api/offline-library/devices/\(id)/retry",
                                                    body: ["version": offlineVersion])
            busy = false; await load(); note = "已按同一版本和快照重试；不会执行动作"
        } catch { note = error.localizedDescription }
    }
    private func review(_ id: String, _ draft: [String: Any], approve: Bool) async {
        guard !busy else { return }; busy = true; defer { busy = false }
        do {
            guard UUID(uuidString: id) != nil, let version = draft["version"] as? Int else {
                throw AdminError.invalidInput("草稿 ID 无效")
            }
            _ = try await model.mobileClient().json("POST", "/api/action-drafts/\(id)/\(approve ? "approve" : "reject")",
                                                    body: ["version": version])
            await load(); note = approve ? "已批准保存；小陆不会立即运动" : "已拒绝草稿"
        } catch { note = error.localizedDescription }
    }
    private func revokeDraft(_ id: String, _ draft: [String: Any]) async {
        guard !busy else { return }; busy = true; defer { busy = false }
        do {
            guard UUID(uuidString: id) != nil, let version = draft["version"] as? Int else {
                throw AdminError.invalidInput("草稿 ID 无效")
            }
            _ = try await model.mobileClient().json("POST", "/api/action-drafts/\(id)/revoke",
                                                    body: ["version": version])
            busy = false; await load(); note = "对应已批准动作已撤销"
        } catch { note = error.localizedDescription }
    }
}

private struct DraftReviewView: View {
    @EnvironmentObject private var model: ParentModel
    let draftID: String
    let onChanged: () async -> Void
    @State private var name = ""
    @State private var steps = [EditedActionStep()]
    @State private var version = 0
    @State private var state = ""
    @State private var source = ""
    @State private var note = ""
    @State private var busy = false
    var body: some View {
        Form {
            Section("孩子的描述") { Text(source).font(.subheadline) }
            Section("编辑草稿") {
                TextField("动作名称", text: $name)
                ForEach($steps) { $step in
                    VStack(alignment: .leading) {
                        Picker("动作", selection: $step.action) {
                            Text("挥手").tag("wave"); Text("坐下").tag("sit")
                            Text("伸懒腰").tag("stretch"); Text("拥抱").tag("hug")
                        }
                        Stepper("完成后等待 \(step.waitMS) 毫秒", value: $step.waitMS, in: 0...3000, step: 100)
                        if steps.count > 1 {
                            Button("删除这一步", role: .destructive) { steps.removeAll { $0.id == step.id } }
                        }
                    }
                }
                Button("添加一步") { if steps.count < 6 { steps.append(EditedActionStep()) } }.disabled(steps.count >= 6)
                Button("保存编辑") { Task { await save() } }.disabled(busy || state != "draft")
                Text("编辑后需重新审阅，再明确批准。批准只保存，不会执行。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("审核状态") {
                Text("\(state) · 版本 \(version)")
                if !note.isEmpty { Text(note).font(.footnote).foregroundStyle(.secondary) }
            }
        }
        .navigationTitle("审核新把戏")
        .task { await load() }
    }
    private func load() async {
        do {
            guard UUID(uuidString: draftID) != nil else { throw AdminError.invalidInput("草稿 ID 无效") }
            let item = try await model.mobileClient().json("GET", "/api/action-drafts/\(draftID)")
            name = item["name"] as? String ?? ""
            source = item["source_text"] as? String ?? ""
            version = item["version"] as? Int ?? 0
            state = item["state"] as? String ?? ""
            steps = ((item["steps"] as? [[String: Any]]) ?? []).map {
                EditedActionStep(action: $0["action"] as? String ?? "wave", waitMS: $0["wait_ms"] as? Int ?? 0)
            }
            if steps.isEmpty { steps = [EditedActionStep()] }
        } catch { note = error.localizedDescription }
    }
    private func save() async {
        guard !busy else { return }; busy = true; defer { busy = false }
        do {
            guard name.range(of: "^[A-Za-z0-9_\\u4e00-\\u9fff-]{1,32}$", options: .regularExpression) != nil,
                  (1...6).contains(steps.count) else { throw AdminError.invalidInput("请检查名称与步骤") }
            let payload = steps.map { ["action": $0.action, "wait_ms": $0.waitMS] as [String: Any] }
            let result = try await model.mobileClient().json("PATCH", "/api/action-drafts/\(draftID)", body: [
                "version": version, "name": name, "steps": payload])
            version = result["version"] as? Int ?? version
            note = "草稿已保存；请再次审核后批准"
            await onChanged()
        } catch { note = error.localizedDescription }
    }
}

private struct GestureManagementView: View {
    @EnvironmentObject private var model: ParentModel
    @State private var note = ""
    @State private var state = ""
    @State private var consent = false
    var body: some View {
        Form {
            Section("短时手势") {
                Text("开启后，小陆会临时拍摄并识别手势，最多 60 秒、最多 3 次动作。请确保现场摆放安全。")
                    .font(.subheadline)
                Toggle("我同意这次拍照和动作", isOn: $consent)
                Button("开启 60 秒") { Task { await call(true) } }.disabled(!consent)
                Button("立即关闭", role: .destructive) { Task { await call(false) } }
                Text("状态：\(state)").font(.footnote).foregroundStyle(.secondary)
            }
            if !note.isEmpty { Section { Text(note).font(.footnote).foregroundStyle(.secondary) } }
        }
        .navigationTitle("手势控制")
        .task { await load() }
    }
    private func load() async {
        do {
            guard let id = devicePath(model) else { throw AdminError.unavailable }
            let result = try await model.mobileClient().json("GET", "/api/gestures/devices/\(id)")
            state = result["enabled"] as? Bool == true ? "已开启" : "已关闭"
        } catch { note = error.localizedDescription }
    }
    private func call(_ enable: Bool) async {
        do {
            guard let id = devicePath(model) else { throw AdminError.unavailable }
            _ = try await model.mobileClient().json("POST", "/api/gestures/devices/\(id)/\(enable ? "enable" : "disable")",
                                                   body: enable ? ["seconds": 60, "consent": true] : nil)
            consent = false
            await load(); note = enable ? "手势识别已短时开启" : "手势识别已关闭"
        } catch { note = error.localizedDescription }
    }
}

private struct GatewayManagementView: View {
    @EnvironmentObject private var model: ParentModel
    @State private var url = ""
    @State private var token = ""
    @State private var version = 0
    @State private var state = ""
    @State private var note = ""
    @State private var change = false
    var body: some View {
        Form {
            Section("小陆连接家中 Mac") {
                TextField("ws://家庭网关:8000/v1/audio", text: $url)
                    .textInputAutocapitalization(.never).autocorrectionDisabled().keyboardType(.URL)
                SecureField("设备令牌（留空沿用当前令牌）", text: $token)
                Text("最近配置：版本 \(version) · \(state)").font(.caption).foregroundStyle(.secondary)
                Text("保存新地址需要小陆重启后才会切换连接。请先确保新地址实际可达。")
                    .font(.caption).foregroundStyle(.secondary)
                Button("保存连接地址") { change = true }.disabled(url.isEmpty)
                Button("按原版本重试保存") { Task { await retry() } }.disabled(version == 0)
                Button("刷新设备配置") { Task { await load() } }
            }
            if !note.isEmpty { Section { Text(note).font(.footnote).foregroundStyle(.secondary) } }
        }
        .navigationTitle("连接与维护")
        .task { await load() }
        .alert("保存小陆的连接地址？", isPresented: $change) {
            Button("保存") { Task { await save() } }
            Button("取消", role: .cancel) { }
        } message: { Text("错误地址可能使设备在下次重启后离线。") }
    }
    private func load() async {
        do {
            guard let id = devicePath(model) else { throw AdminError.unavailable }
            let result = try await model.mobileClient().json("GET", "/api/gateway/devices/\(id)")
            let latest = result["latest"] as? [String: Any]
            url = latest?["url"] as? String ?? url
            version = latest?["version"] as? Int ?? 0
            state = latest?["state"] as? String ?? "尚未设置"
            note = "已读取设备连接配置"
        } catch { note = error.localizedDescription }
    }
    private func save() async {
        do {
            guard let id = devicePath(model),
                  let parts = URLComponents(string: url), ["ws", "wss"].contains(parts.scheme ?? ""),
                  parts.host != nil, parts.user == nil, parts.password == nil else {
                throw AdminError.invalidInput("请输入有效的 ws:// 或 wss:// 网关地址")
            }
            var fields: [String: Any] = ["url": url]
            if !token.isEmpty { fields["token"] = token }
            _ = try await model.mobileClient().json("POST", "/api/gateway/devices/\(id)/configure", body: fields)
            token = ""; await load(); note = "连接地址已保存；设备重启后生效"
        } catch { token = ""; note = error.localizedDescription }
    }
    private func retry() async {
        do {
            guard let id = devicePath(model), version > 0 else { throw AdminError.unavailable }
            _ = try await model.mobileClient().json("POST", "/api/gateway/devices/\(id)/retry",
                                                    body: ["version": version])
            await load(); note = "已按同一版本重试；请核对设备回执"
        } catch { note = error.localizedDescription }
    }
}

private struct TextChatView: View {
    @EnvironmentObject private var model: ParentModel
    @State private var member = ""
    @State private var mode = "chat"
    @State private var prompt = ""
    @State private var replies: [(String, String)] = []
    @State private var sessionID = UUID().uuidString.lowercased()
    @State private var busy = false
    @State private var note = ""

    var body: some View {
        Form {
            Section("对话对象") {
                TextField("家长确认的成员 ID；留空则匿名", text: $member)
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                Picker("模式", selection: $mode) {
                    Text("聊天").tag("chat"); Text("英语").tag("english"); Text("故事").tag("story")
                }
                Text("文字对话由家长手动选择成员。语音对话仍依赖小陆识别说话人。请在现场使用：自然语言也可能触发已批准动作。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("和小陆说话") {
                TextField("输入一句话", text: $prompt, axis: .vertical).lineLimit(2...4)
                Button(busy ? "小陆正在回答…" : "发送") { Task { await send() } }
                    .disabled(busy || prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            if !replies.isEmpty {
                Section("本次对话") {
                    ForEach(replies.indices, id: \.self) { index in
                        VStack(alignment: .leading, spacing: 6) {
                            Text("你：\(replies[index].0)").font(.subheadline)
                            Text("小陆：\(replies[index].1)").font(.body.weight(.medium))
                        }
                    }
                }
            }
            if !note.isEmpty { Section { Text(note).font(.footnote).foregroundStyle(.secondary) } }
        }
        .navigationTitle("文字陪伴")
    }

    private func send() async {
        guard !busy else { return }; busy = true; defer { busy = false }
        do {
            guard let id = devicePath(model), member.isEmpty || safeMember(member) else {
                throw AdminError.invalidInput("成员 ID 不正确")
            }
            let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
            guard text.count <= 300 else { throw AdminError.invalidInput("每次最多输入 300 字") }
            let result = try await model.mobileClient().json("POST", "/api/chat", body: [
                "device_id": id, "text": text, "mode": mode, "session_id": sessionID,
                "person_id": member.isEmpty ? NSNull() : member])
            let reply = result["reply"] as? String ?? "没有得到文字回答"
            replies.append((text, reply)); prompt = ""
            if let receipt = result["device_receipt"] as? [String: Any] {
                note = receipt["ok"] as? Bool == true ? "关联动作已由设备确认" : "关联动作未完成"
            } else { note = "家中 Mac 已回复" }
        } catch { note = error.localizedDescription }
    }
}

private struct PhotoGalleryView: View {
    @EnvironmentObject private var model: ParentModel
    @State private var photos: [[String: Any]] = []
    @State private var selectedImage: UIImage?
    @State private var note = ""
    @State private var busy = false
    var body: some View {
        Form {
            Section("最近拍照") {
                if photos.isEmpty { Text("暂无照片").foregroundStyle(.secondary) }
                ForEach(photos.indices, id: \.self) { index in
                    let photo = photos[index]
                    let id = photo["photo_id"] as? String ?? ""
                    Button {
                        Task { await show(id) }
                    } label: {
                        HStack {
                            Label("查看照片", systemImage: "photo")
                            Spacer()
                            if let time = photo["captured_at"] as? Double {
                                Text(Date(timeIntervalSince1970: time).formatted(date: .abbreviated, time: .shortened))
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }.disabled(busy || UUID(uuidString: id) == nil)
                }
                Button("刷新照片") { Task { await load() } }.disabled(busy)
            }
            if let selectedImage {
                Section("本机照片") {
                    Image(uiImage: selectedImage).resizable().scaledToFit()
                    Text("此处只显示已保存的单张照片，不代表实时画面或全屋巡视。")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            if !note.isEmpty { Section { Text(note).font(.footnote).foregroundStyle(.secondary) } }
        }
        .navigationTitle("最近照片")
        .task { await load() }
    }
    private func load() async {
        do {
            guard let id = devicePath(model) else { throw AdminError.unavailable }
            photos = try await model.mobileClient().jsonArray("GET", "/api/photos/\(id)")
            note = "已读取家中 Mac 保存的照片列表"
        } catch { note = error.localizedDescription }
    }
    private func show(_ id: String) async {
        guard !busy else { return }; busy = true; defer { busy = false }
        do {
            let data = try await model.mobileClient().request("GET", "/api/photo/\(id)")
            guard let image = UIImage(data: data), image.size.width <= 1920, image.size.height <= 1920 else {
                throw AdminError.invalidResponse
            }
            selectedImage = image; note = "照片已从家中 Mac 加密读取"
        } catch { note = error.localizedDescription }
    }
}

private struct EnglishCheckView: View {
    @EnvironmentObject private var model: ParentModel
    @State private var member = ""
    @State private var word = "dog"
    @State private var checkpoint = "baseline"
    @State private var result = "not_yet"
    @State private var modality = "spoken"
    @State private var checks: [[String: Any]] = []
    @State private var note = ""
    var body: some View {
        Form {
            Section("家长观察") {
                TextField("成员 ID", text: $member).textInputAutocapitalization(.never).autocorrectionDisabled()
                TextField("英文词", text: $word).textInputAutocapitalization(.never).autocorrectionDisabled()
                Picker("时间", selection: $checkpoint) {
                    Text("学习前").tag("baseline"); Text("隔天").tag("next_day"); Text("一周后").tag("week_later")
                }
                Picker("结果", selection: $result) {
                    Text("独立做到").tag("independent"); Text("提醒后做到").tag("prompted"); Text("暂时没做到").tag("not_yet")
                }
                Picker("观察方式", selection: $modality) {
                    Text("指认").tag("pointed"); Text("说出").tag("spoken"); Text("在情境中使用").tag("used_in_context")
                }
                Button("保存观察") { Task { await save() } }.disabled(!safeMember(member))
                Text("这些是家长观察，不等于小陆已验证孩子发音或掌握程度。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("已记录") {
                Button("读取记录") { Task { await load() } }.disabled(!safeMember(member))
                ForEach(checks.indices, id: \.self) { index in
                    let item = checks[index]
                    Text("\(item["word"] as? String ?? "") · \(item["checkpoint"] as? String ?? "") · \(item["result"] as? String ?? "")")
                }
            }
            if !note.isEmpty { Section { Text(note).font(.footnote).foregroundStyle(.secondary) } }
        }
        .navigationTitle("英语观察")
    }
    private func path() throws -> String {
        guard let id = devicePath(model), safeMember(member) else { throw AdminError.invalidInput("请输入成员 ID") }
        return "/api/memory/\(id)/\(member)"
    }
    private func load() async {
        do {
            let state = try await model.mobileClient().json("GET", path())
            checks = state["english_checks"] as? [[String: Any]] ?? []
            note = "已读取家长观察记录"
        } catch { note = error.localizedDescription }
    }
    private func save() async {
        do {
            let path = try path()
            _ = try await model.mobileClient().json("POST", path + "/english-checks", body: [
                "word": word.lowercased(), "checkpoint": checkpoint, "result": result, "modality": modality])
            await load(); note = "家长观察已保存"
        } catch { note = error.localizedDescription }
    }
}

private struct RoutineManagementView: View {
    @EnvironmentObject private var model: ParentModel
    @State private var member = ""
    @State private var routines: [[String: Any]] = []
    @State private var routineID = ""
    @State private var title = ""
    @State private var steps = ""
    @State private var note = ""
    @State private var busy = false
    @State private var delete = false

    var body: some View {
        Form {
            Section("家人") {
                TextField("已登记成员 ID", text: $member)
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                Button("读取生活小任务") { Task { await load() } }.disabled(busy || !safeMember(member))
            }
            Section("选择或新建") {
                Picker("已有任务", selection: $routineID) {
                    Text("新建任务").tag("")
                    ForEach(routines.indices, id: \.self) { index in
                        let item = routines[index]
                        if let id = item["routine_id"] as? String {
                            Text(item["title"] as? String ?? id).tag(id)
                        }
                    }
                }
                .onChange(of: routineID) { _, id in select(id) }
                TextField("任务 ID，如 tidy-toys", text: $routineID)
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                TextField("任务名称，如收玩具", text: $title)
                Text("每行一步，最多 8 步；这里只记录孩子自己报告的进度。")
                    .font(.caption).foregroundStyle(.secondary)
                TextEditor(text: $steps).frame(minHeight: 110)
                Button("保存文字步骤并重置进度") { Task { await save() } }
                    .disabled(busy || !safeMember(member) || !safeMember(routineID))
            }
            if let selected = selected {
                Section("当前进度") {
                    Text("状态：\(selected["state"] as? String ?? "未知") · \(selected["step_index"] as? Int ?? 0) / \((selected["steps"] as? [String])?.count ?? 0)")
                    if let current = selected["current_step"] as? String { Text(current) }
                    HStack {
                        Button("开始") { Task { await transition("start") } }
                        Button("完成这一步") { Task { await transition("advance") } }
                    }.disabled(busy)
                    HStack {
                        Button("暂停") { Task { await transition("pause") } }
                        Button("继续") { Task { await transition("resume") } }
                        Button("重置") { Task { await transition("reset") } }
                    }.disabled(busy)
                    Button("删除此任务", role: .destructive) { delete = true }
                }
            }
            if !note.isEmpty { Section { Text(note).font(.footnote).foregroundStyle(.secondary) } }
        }
        .navigationTitle("生活小任务")
        .alert("删除生活任务？", isPresented: $delete) {
            Button("删除", role: .destructive) { Task { await remove() } }
            Button("取消", role: .cancel) { }
        } message: { Text("该任务的文字步骤和当前进度都会删除。") }
    }

    private var selected: [String: Any]? {
        routines.first { $0["routine_id"] as? String == routineID }
    }
    private func path() throws -> String {
        guard let id = devicePath(model), safeMember(member) else { throw AdminError.invalidInput("请输入有效家人成员 ID") }
        return "/api/routines/\(id)/members/\(member)"
    }
    private func select(_ id: String) {
        if let item = routines.first(where: { $0["routine_id"] as? String == id }) {
            title = item["title"] as? String ?? ""
            steps = (item["steps"] as? [String] ?? []).joined(separator: "\n")
        } else { title = ""; steps = "" }
    }
    private func load() async {
        guard !busy else { return }; busy = true; defer { busy = false }
        do {
            let result = try await model.mobileClient().json("GET", path())
            routines = result["routines"] as? [[String: Any]] ?? []
            select(routineID)
            note = "已读取此人的生活任务"
        } catch { note = error.localizedDescription }
    }
    private func save() async {
        guard !busy else { return }; busy = true; defer { busy = false }
        do {
            guard safeMember(routineID), !title.isEmpty, title.count <= 40 else {
                throw AdminError.invalidInput("请填写有效任务 ID 和名称")
            }
            let lines = steps.split(separator: "\n", omittingEmptySubsequences: false)
                .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            guard (1...8).contains(lines.count), lines.allSatisfy({ !$0.isEmpty && $0.count <= 80 }) else {
                throw AdminError.invalidInput("请输入 1–8 行文字，每行不超过 80 字")
            }
            let revision = selected?["revision"] as? Int ?? 0
            _ = try await model.mobileClient().json("PUT", path() + "/" + routineID, body: [
                "title": title, "steps": lines, "expected_revision": revision])
            busy = false; await load(); note = "任务步骤已保存，进度已重置"
        } catch { note = error.localizedDescription }
    }
    private func transition(_ operation: String) async {
        guard !busy else { return }; busy = true; defer { busy = false }
        do {
            guard let item = selected, let revision = item["revision"] as? Int,
                  let index = item["step_index"] as? Int else { throw AdminError.invalidInput("请先读取一个任务") }
            _ = try await model.mobileClient().json("POST", path() + "/" + routineID + "/transition", body: [
                "operation": operation, "expected_revision": revision, "expected_step_index": index])
            busy = false; await load(); note = "任务进度已更新；由家长或孩子自行确认完成"
        } catch { note = error.localizedDescription }
    }
    private func remove() async {
        guard !busy else { return }; busy = true; defer { busy = false }
        do {
            guard let revision = selected?["revision"] as? Int else { throw AdminError.invalidInput("请先读取一个任务") }
            _ = try await model.mobileClient().json("DELETE", path() + "/" + routineID,
                                                    body: ["expected_revision": revision])
            routineID = ""; busy = false; await load(); note = "任务已删除"
        } catch { note = error.localizedDescription }
    }
}
