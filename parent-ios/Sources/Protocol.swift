import Foundation
import CryptoKit
import Security

enum ClientError: LocalizedError {
    case invalid(String), downloading, busy, timeout
    var errorDescription: String? {
        switch self {
        case .invalid(let message): return message
        case .downloading: return "iCloud 正在下载，请稍后刷新"
        case .busy: return "上一次 iCloud 访问仍在等待，暂不重复请求"
        case .timeout: return "iCloud 暂未响应；已有记录已保留"
        }
    }
}

struct Pairing: Codable, Equatable {
    var version: Int = 1
    let serverID: String
    let deviceID: String
    let serverPublicKey: String
    var adminURL: String? = nil
    var adminCACertificate: String? = nil
    var adminAttestation: String? = nil
    enum CodingKeys: String, CodingKey {
        case version, serverID = "server_id", deviceID = "device_id", serverPublicKey = "server_public_key",
             adminURL = "admin_url", adminCACertificate = "admin_ca_certificate",
             adminAttestation = "admin_attestation"
    }
    func validate() throws {
        guard version == 1, Self.validID(serverID), Self.validID(deviceID),
              let raw = Data(base64Encoded: serverPublicKey), raw.count == 32,
              (try? Curve25519.Signing.PublicKey(rawRepresentation: raw)) != nil else {
            throw ClientError.invalid("配对配置无效，请从家中 Mac 重新导出")
        }
        if adminURL != nil || adminCACertificate != nil || adminAttestation != nil {
            guard let adminURL, let adminCACertificate, let adminAttestation,
                  let url = URLComponents(string: adminURL), url.scheme == "https",
                  let host = url.host, !host.isEmpty, url.user == nil, url.password == nil,
                  url.query == nil, url.fragment == nil, url.path.isEmpty,
                  let certificate = Data(base64Encoded: adminCACertificate),
                  SecCertificateCreateWithData(nil, certificate as CFData) != nil,
                  let signature = Data(base64Encoded: adminAttestation), signature.count == 64,
                  let key = try? Curve25519.Signing.PublicKey(rawRepresentation: raw) else {
                throw ClientError.invalid("手机管理连接配置无效，请重新导入")
            }
            let statement = "PUPPY-ADMIN-CONFIG-V1\n\(serverID)\n\(deviceID)\n\(adminURL)\n\(Wire.hash(certificate))"
            guard key.isValidSignature(signature, for: Data(statement.utf8)) else {
                throw ClientError.invalid("手机管理连接未得到家庭服务器签名")
            }
        }
    }
    static func validID(_ value: String) -> Bool {
        value.range(of: "^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$", options: .regularExpression) != nil
    }
    var fingerprint: String {
        guard let data = Data(base64Encoded: serverPublicKey) else { return "—" }
        return String(Wire.hash(data).prefix(16))
    }
}

enum CommandKind: String, Codable, CaseIterable {
    case status, photo, patrolOnce = "patrol_once", stop, action
    var title: String {
        switch self {
        case .status: return "查看状态"
        case .photo: return "拍张照片"
        case .patrolOnce: return "定点巡看"
        case .stop: return "请求停止"
        case .action: return "短动作"
        }
    }
}

struct Command: Codable, Equatable, Identifiable {
    let version: Int
    let requestID: String
    let sourceID: String
    let serverID: String
    let deviceID: String
    let kind: CommandKind
    let params: [String: String]
    let createdAt: Int
    let expiresAt: Int
    var id: String { requestID }
    enum CodingKeys: String, CodingKey {
        case version, kind, params
        case requestID = "request_id", sourceID = "source_id", serverID = "server_id", deviceID = "device_id"
        case createdAt = "created_at", expiresAt = "expires_at"
    }
    static let safeActions = ["wave", "sit", "stretch", "hug", "stop"]
    static func make(kind: CommandKind, action: String? = nil, pairing: Pairing, sourceID: String,
                     now: Date = Date(), ttl: Int = 300) throws -> Command {
        try pairing.validate()
        guard Pairing.validID(sourceID), (1...300).contains(ttl),
              (kind == .action ? action.map(Self.safeActions.contains) == true : action == nil) else {
            throw ClientError.invalid("不支持的命令或有效期")
        }
        let created = Int(now.timeIntervalSince1970)
        return Command(version: 1, requestID: UUID().uuidString.lowercased(), sourceID: sourceID,
                       serverID: pairing.serverID, deviceID: pairing.deviceID, kind: kind,
                       params: action.map { ["name": $0] } ?? [:], createdAt: created, expiresAt: created + ttl)
    }
}

struct Envelope: Codable {
    let payload: String
    let signature: String
}

enum Wire {
    static func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    static func encode<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(value)
    }
    static func sign<T: Encodable>(_ value: T, key: Curve25519.Signing.PrivateKey) throws -> Data {
        let bytes = try encode(value)
        return try encode(Envelope(payload: bytes.base64EncodedString(), signature: key.signature(for: bytes).base64EncodedString()))
    }
    static func verified<T: Decodable>(_ data: Data, publicKey: String, as type: T.Type) throws -> T {
        guard data.count <= 131_072, let keyData = Data(base64Encoded: publicKey) else {
            throw ClientError.invalid("回执过大或公钥无效")
        }
        let envelope = try JSONDecoder().decode(Envelope.self, from: data)
        guard let payload = Data(base64Encoded: envelope.payload), let signature = Data(base64Encoded: envelope.signature),
              signature.count == 64,
              try Curve25519.Signing.PublicKey(rawRepresentation: keyData).isValidSignature(signature, for: payload) else {
            throw ClientError.invalid("签名校验失败，未采信这份数据")
        }
        // Verify the exact transmitted bytes, never re-encoded JSON.
        return try JSONDecoder().decode(type, from: payload)
    }
}

struct Battery: Codable, Equatable {
    let charging: Bool?
    let level: Int?
    let chargingKnown: Bool?
    let servoVoltageV: Double?

    init(charging: Bool? = nil, level: Int? = nil, chargingKnown: Bool? = nil, servoVoltageV: Double? = nil) {
        self.charging = charging
        self.level = level
        self.chargingKnown = chargingKnown
        self.servoVoltageV = servoVoltageV
    }

    enum CodingKeys: String, CodingKey {
        case charging, level, chargingKnown = "charging_known", servoVoltageV = "servo_voltage_v"
    }
}
struct DeviceStatus: Codable, Equatable {
    let online: Bool?
    let observedAt: Double?
    let volume: Int?
    let battery: Battery?
    let petName: String?
    enum CodingKeys: String, CodingKey {
        case online, volume, battery, observedAt = "observed_at", petName = "pet_name"
    }
    func fresh(at now: Date) -> Bool {
        guard let observedAt else { return false }
        let age = now.timeIntervalSince1970 - observedAt
        return age >= -30 && age <= 60
    }
}

struct Snapshot: Codable, Equatable {
    let version: Int
    let serverID: String
    let deviceID: String
    let publishedAt: Double
    let status: DeviceStatus
    enum CodingKeys: String, CodingKey {
        case version, status, serverID = "server_id", deviceID = "device_id", publishedAt = "published_at"
    }
    func validate(pairing: Pairing, now: Date = Date()) throws {
        guard version == 1, serverID == pairing.serverID, deviceID == pairing.deviceID,
              publishedAt <= now.timeIntervalSince1970 + 30 else {
            throw ClientError.invalid("状态来源或时间不匹配")
        }
    }
    func online(at now: Date) -> Bool {
        status.online == true && status.fresh(at: now) && now.timeIntervalSince1970 - publishedAt <= 60
    }
}

struct MediaReference: Codable, Equatable {
    let file: String
    let sha256: String
    let bytes: Int
    func validate(_ data: Data, requestID: String) throws {
        guard file == "media/\(requestID).jpg", (1...512_000).contains(bytes), data.count == bytes,
              sha256 == Wire.hash(data) else { throw ClientError.invalid("照片校验失败，未显示") }
    }
}

struct PhotoDescription: Codable, Equatable {
    let photoID: String?
    let reply: String?
    let uncertain: Bool?
    let scope: String?
    let ok: Bool?
    let error: String?
    enum CodingKeys: String, CodingKey { case photoID = "photo_id", reply, uncertain, scope, ok, error }
    var displayReply: String? {
        guard ok != false, scope == "single_photo", let reply else { return nil }
        let text = reply.trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? nil : text
    }
}

struct ReceiptResult: Codable, Equatable {
    let ok: Bool?
    let status: DeviceStatus?
    let media: MediaReference?
    let capturedAt: Double?
    let error: String?
    var description: PhotoDescription? = nil
    enum CodingKeys: String, CodingKey { case ok, status, media, error, description, capturedAt = "captured_at" }
}

enum ReceiptState: String, Codable {
    case accepted, executing, succeeded, failed, rejected, unknown
    var terminal: Bool { [.succeeded, .failed, .rejected, .unknown].contains(self) }
    var title: String {
        switch self {
        case .accepted: return "已接收"
        case .executing: return "执行中"
        case .succeeded: return "已完成"
        case .failed: return "执行失败"
        case .rejected: return "已拒绝"
        case .unknown: return "结果待核实"
        }
    }
}

struct Receipt: Codable, Equatable {
    let version: Int
    let requestID: String
    let serverID: String
    let sourceID: String?
    let deviceID: String?
    let kind: String?
    let state: ReceiptState
    let updatedAt: Double
    let result: ReceiptResult?
    let error: String?
    enum CodingKeys: String, CodingKey {
        case version, kind, state, result, error
        case requestID = "request_id", serverID = "server_id", sourceID = "source_id", deviceID = "device_id", updatedAt = "updated_at"
    }
    var isUnboundRejection: Bool { state == .rejected && sourceID == nil && deviceID == nil && kind == nil }
    func validate(command: Command, now: Date = Date(), requireFullTarget: Bool = false) throws {
        guard version == 1, requestID == command.requestID, serverID == command.serverID,
              updatedAt >= Double(command.createdAt) - 30, updatedAt <= now.timeIntervalSince1970 + 30 else {
            throw ClientError.invalid("回执与请求不匹配")
        }
        // Authenticated rejections in protocol v1 intentionally omit source and
        // device fields. Every receipt claiming execution must bind all fields.
        if requireFullTarget || !isUnboundRejection {
            guard sourceID == command.sourceID, deviceID == command.deviceID, kind == command.kind.rawValue else {
                throw ClientError.invalid("回执目标不匹配")
            }
        }
        if state == .succeeded && result?.ok != true { throw ClientError.invalid("成功回执缺少执行确认") }
    }
    func supersedes(_ previous: Receipt?) throws -> Bool {
        guard let previous else { return true }
        if previous.state.terminal {
            guard self == previous || !state.terminal else { throw ClientError.invalid("收到冲突回执，需核实实际状态") }
            return false
        }
        if updatedAt < previous.updatedAt { return false }
        if previous.state == .executing && state == .accepted { return false }
        return self != previous
    }
}

struct RequestRecord: Codable, Identifiable {
    let command: Command
    var publicationConfirmed: Bool
    var receipt: Receipt?
    var rejectionNotice: Receipt? = nil
    var id: String { command.requestID }
    mutating func separateLegacyRejection() {
        if let receipt, receipt.isUnboundRejection {
            rejectionNotice = receipt
            self.receipt = nil
        }
    }
    func title(now: Date = Date()) -> String {
        if receipt?.state == .rejected && receipt?.error == "expired" { return "已过期 · 服务端拒绝" }
        if let receipt {
            if !receipt.state.terminal && now.timeIntervalSince1970 >= Double(command.expiresAt) {
                return "已过期 · 结果待核实"
            }
            return receipt.state.title
        }
        if now.timeIntervalSince1970 >= Double(command.expiresAt) { return "已过期 · 等待回执" }
        return publicationConfirmed ? "已提交" : "提交待确认"
    }
}
