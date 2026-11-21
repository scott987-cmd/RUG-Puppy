import Foundation
import SwiftUI
import CryptoKit

@MainActor
final class ParentModel: ObservableObject {
    @Published var pairing: Pairing?
    @Published var directory: URL?
    @Published var records: [RequestRecord] = []
    @Published var snapshot: Snapshot?
    @Published var photos: [String: UIImage] = [:]
    @Published var unavailablePhotos: Set<String> = []
    @Published var message: String?
    @Published var refreshing = false
    @Published var sending = false
    @Published var lastRefresh: Date?
    @Published var adminConnectionMessage: String?
    let isDemo: Bool
    private(set) var identity: PhoneIdentity?
    private let io = DirectoryIO()
    private let defaults = UserDefaults.standard
    private let historyURL: URL
    private var connectionGeneration = UUID()
    private var adminCheckedSignature: String?
    var ready: Bool { pairing != nil && directory != nil && identity != nil }
    var registration: String { (try? identity?.registration(deviceID: pairing?.deviceID ?? "请先导入配置")) ?? "" }
    func mobileClient() throws -> MobileAdminClient {
        guard !isDemo, let pairing, let identity else { throw AdminError.unavailable }
        return try MobileAdminClient(pairing: pairing, identity: identity)
    }
    func checkMobileAdmin() async {
        guard !isDemo, let signature = pairing?.adminAttestation else { return }
        adminCheckedSignature = signature
        do {
            _ = try await mobileClient().json("GET", "/api/status")
            adminConnectionMessage = "已通过手机签名连接家中 Mac"
        } catch { adminConnectionMessage = error.localizedDescription }
    }
    var shouldCheckMobileAdmin: Bool {
        !isDemo && pairing?.adminAttestation != nil && pairing?.adminAttestation != adminCheckedSignature
    }

    init(demo: Bool = false) {
        #if DEBUG
        isDemo = demo || ProcessInfo.processInfo.arguments.contains("--ui-demo")
        #else
        isDemo = demo
        #endif
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        historyURL = support.appendingPathComponent("parent-requests.json")
        do {
            if isDemo {
                identity = PhoneIdentity(privateKey: .init())
                try seedDemo(stale: ProcessInfo.processInfo.arguments.contains("--stale"))
                return
            }
            identity = try PhoneIdentity.load()
            if let data = defaults.data(forKey: "pairing-v1") {
                let value = try JSONDecoder().decode(Pairing.self, from: data)
                try value.validate()
                pairing = value
            }
            if let bookmark = defaults.data(forKey: "directory-bookmark-v1") {
                var stale = false
                let url = try URL(resolvingBookmarkData: bookmark, options: [.withoutUI], bookmarkDataIsStale: &stale)
                if stale { message = "目录授权需要更新，请重新选择 PuppyCompanion" }
                else if url.lastPathComponent == "PuppyCompanion" { directory = url }
            }
            if let data = try? Data(contentsOf: historyURL) {
                records = (try? JSONDecoder().decode([RequestRecord].self, from: data)) ?? []
                for index in records.indices { records[index].separateLegacyRejection() }
            }
        } catch { message = error.localizedDescription }
    }

    func importPairing(_ url: URL) async {
        guard !isDemo else { message = "演示使用隔离配置"; return }
        do {
            let value: Pairing = try await io.run { budget in
                let value = try JSONDecoder().decode(Pairing.self, from: DirectoryIO.selectedFileData(url, maximum: 8192, budget: budget))
                try value.validate()
                return value
            }
            if pairing != value { records = []; snapshot = nil; photos = [:]; unavailablePhotos = [] }
            adminCheckedSignature = nil
            connectionGeneration = UUID()
            pairing = value
            defaults.set(try Wire.encode(value), forKey: "pairing-v1")
            persist()
            message = "配置已导入，请核对服务器指纹并在 Mac 登记手机公钥"
        } catch { message = error.localizedDescription }
    }

    func selectDirectory(_ url: URL) async {
        guard !isDemo else { message = "演示不会连接真实 iCloud 目录"; return }
        do {
            guard url.lastPathComponent == "PuppyCompanion" else { throw ClientError.invalid("请选择名为 PuppyCompanion 的共享目录") }
            let bookmark: Data = try await io.run { budget in
                try budget.check()
                return try DirectoryIO.scoped(url) {
                    try url.bookmarkData(options: [.minimalBookmark], includingResourceValuesForKeys: nil, relativeTo: nil)
                }
            }
            defaults.set(bookmark, forKey: "directory-bookmark-v1")
            connectionGeneration = UUID()
            directory = url
            message = "目录已连接。登记手机公钥后，可发送状态请求"
        } catch { message = error.localizedDescription }
    }

    func refresh() async {
        guard !refreshing, !sending, let pairing, let directory else { return }
        refreshing = true
        defer { refreshing = false }
        let known = records
        let verifiedHashes = Dictionary(uniqueKeysWithValues: known.compactMap { record -> (String, String)? in
            guard photos[record.id] != nil, let hash = record.receipt?.result?.media?.sha256 else { return nil }
            return (record.id, hash)
        })
        let generation = connectionGeneration
        do {
            let (update, upgradedPairing) = try await io.run { budget in
                try DirectoryIO.scoped(directory) {
                    let update = try DirectoryReader.read(root: directory, pairing: pairing, records: known,
                                                          budget: budget, verifiedMediaHashes: verifiedHashes)
                    let publicFile = directory.appendingPathComponent("home-macbook.json")
                    let candidate = try? DirectoryIO.availableData(publicFile, maximum: 8192,
                                                                    root: directory, budget: budget)
                    return (update, candidate)
                }
            }
            guard generation == connectionGeneration else { return }
            if let upgradedPairing, let candidate = try? JSONDecoder().decode(Pairing.self, from: upgradedPairing),
               (try? candidate.validate()) != nil, candidate.serverID == pairing.serverID,
               candidate.deviceID == pairing.deviceID, candidate.serverPublicKey == pairing.serverPublicKey,
               candidate.adminURL != nil, candidate != pairing {
                self.pairing = candidate
                adminCheckedSignature = nil
                defaults.set(try Wire.encode(candidate), forKey: "pairing-v1")
            }
            if let incoming = update.snapshot, incoming.publishedAt >= (snapshot?.publishedAt ?? 0) { snapshot = incoming }
            for index in records.indices {
                if let receipt = update.receipts[records[index].id] { records[index].receipt = receipt }
                if let notice = update.rejectionNotices[records[index].id] { records[index].rejectionNotice = notice }
            }
            unavailablePhotos = update.unavailableMediaIDs
            for id in unavailablePhotos { photos.removeValue(forKey: id) }
            for (id, bytes) in update.photos {
                if let image = UIImage(data: bytes), let cg = image.cgImage, cg.width <= 640, cg.height <= 640 {
                    photos[id] = image
                } else { message = "照片格式或尺寸不支持" }
            }
            if !update.notes.isEmpty { message = Array(Set(update.notes)).sorted().prefix(2).joined(separator: "；") }
            else { message = nil }
            lastRefresh = Date()
            persist()
        } catch { message = error.localizedDescription }
    }

    func send(_ kind: CommandKind, action: String? = nil) async {
        guard !sending, let pairing, let directory, let identity else { return }
        guard !isDemo || kind == .status else { message = "演示模式仅允许向隔离目录提交状态请求"; return }
        sending = true
        defer { sending = false }
        do {
            let command = try Command.make(kind: kind, action: action, pairing: pairing, sourceID: identity.sourceID)
            let signed = try Wire.sign(command, key: identity.privateKey)
            // Keep the ID before touching iCloud. A timed-out write may finish
            // later; never automatically repeat it under a new request ID.
            records.insert(RequestRecord(command: command, publicationConfirmed: false, receipt: nil), at: 0)
            records = Array(records.prefix(100))
            guard persist() else {
                records.removeAll { $0.id == command.id }
                message = "无法保存请求记录，本次未提交"
                return
            }
            try await io.run { budget in try DirectoryIO.publish(signed, root: directory, command: command, budget: budget) }
            if let index = records.firstIndex(where: { $0.id == command.id }) { records[index].publicationConfirmed = true }
            persist()
            message = "已提交，等待家中 Mac 接收"
        } catch { message = "提交尚未确认：\(error.localizedDescription)。请在记录中等待回执" }
    }

    @discardableResult
    private func persist() -> Bool {
        guard !isDemo else { return true }
        do {
            try FileManager.default.createDirectory(at: historyURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Wire.encode(records).write(to: historyURL, options: [.atomic, .completeFileProtection])
            return true
        } catch {
            message = "手机记录保存失败，请勿重复提交：\(error.localizedDescription)"
            return false
        }
    }

    private func seedDemo(stale: Bool) throws {
        let serverKey = Curve25519.Signing.PrivateKey()
        let pairing = Pairing(serverID: "demo-home-mac", deviceID: "demo-puppy", serverPublicKey: serverKey.publicKey.rawRepresentation.base64EncodedString())
        self.pairing = pairing
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathComponent("PuppyCompanion")
        directory = root
        for folder in ["status", "receipts", "commands", "media"] {
            try FileManager.default.createDirectory(at: root.appendingPathComponent(folder), withIntermediateDirectories: true)
        }
        let now = Date().timeIntervalSince1970
        let observed = now - (stale ? 3600 : 5)
        let status = DeviceStatus(online: true, observedAt: observed, volume: 0, battery: Battery(charging: false, level: 86), petName: "豆豆")
        let snapshot = Snapshot(version: 1, serverID: pairing.serverID, deviceID: pairing.deviceID, publishedAt: observed, status: status)
        try Wire.sign(snapshot, key: serverKey).write(to: root.appendingPathComponent("status/demo-puppy.json"))
        let command = try Command.make(kind: .photo, pairing: pairing, sourceID: identity!.sourceID, now: Date(timeIntervalSince1970: now - 12))
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: 240, height: 240), format: format)
        let bytes = renderer.image { context in
            UIColor(red: 0.87, green: 0.94, blue: 0.92, alpha: 1).setFill()
            context.fill(CGRect(x: 0, y: 0, width: 240, height: 240))
            let text = "🐾\n隔离演示照片" as NSString
            text.draw(in: CGRect(x: 25, y: 75, width: 210, height: 110), withAttributes: [.font: UIFont.systemFont(ofSize: 26), .foregroundColor: UIColor.darkGray])
        }.jpegData(compressionQuality: 0.8)!
        let media = MediaReference(file: "media/\(command.id).jpg", sha256: Wire.hash(bytes), bytes: bytes.count)
        let arguments = ProcessInfo.processInfo.arguments
        let description: PhotoDescription? = arguments.contains("--no-description") ? nil :
            arguments.contains("--description-unavailable") ?
            PhotoDescription(photoID: nil, reply: nil, uncertain: nil, scope: nil, ok: false, error: "description_unavailable") :
            PhotoDescription(photoID: "demo-photo", reply: "照片里能看到浅绿色背景和爪印，无法判断房间其他位置。",
                             uncertain: true, scope: "single_photo", ok: nil, error: nil)
        let result = ReceiptResult(ok: true, status: nil, media: media, capturedAt: now - 8, error: nil, description: description)
        let receipt = Receipt(version: 1, requestID: command.id, serverID: pairing.serverID, sourceID: command.sourceID,
                              deviceID: pairing.deviceID, kind: "photo", state: .succeeded, updatedAt: now - 3, result: result, error: nil)
        records = [RequestRecord(command: command, publicationConfirmed: true, receipt: nil)]
        if !arguments.contains("--missing-photo") {
            try bytes.write(to: root.appendingPathComponent(media.file))
        }
        try Wire.sign(receipt, key: serverKey).write(to: root.appendingPathComponent("receipts/\(command.id).json"))
    }
}
