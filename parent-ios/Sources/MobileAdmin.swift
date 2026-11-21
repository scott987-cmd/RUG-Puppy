import Foundation
import CryptoKit
import Security

enum AdminError: LocalizedError {
    case unavailable, invalidResponse, invalidInput(String), server(Int, String), trust, network
    private static let identityHints: [String: String] = [
        "voice_too_quiet": "声音太小。请重新开始录音，并从一开始连续说话 5–8 秒。",
        "insufficient_speech": "有效说话时间不足。请重新录制，并连续说话 5–8 秒。",
        "voice_clipped": "声音过大或失真。请稍离开麦克风再录。",
        "no_face": "照片里没有检测到人脸。请让本人正脸完整入镜、避开逆光后重拍。",
        "multiple_faces": "照片里有多张人脸。请只拍要录入的本人。",
        "face_too_small": "人脸太小。请靠近一些再拍。",
        "face_clipped": "人脸被画面边缘截断。请完整入镜再拍。",
        "face_blurred": "照片模糊。请拿稳手机后重拍。",
        "face_lighting": "光线不合适。请避开逆光或过暗环境重拍。",
        "face_pose": "请让本人正对摄像头后重拍。",
        "enrollment_inconsistent_retake_sample": "本次样本与此前录入的本人不一致，请确认是同一位家人后重试。",
        "duplicate_sample": "这份样本与已有样本过于相似，请换一句话或角度重录。",
        "sample_limit_revoke_one_first": "该成员已达到样本上限，请先删除一份旧样本。"
    ]
    var errorDescription: String? {
        switch self {
        case .unavailable: return "请先导入带手机管理连接的家庭配对配置，并连接家中 Wi-Fi"
        case .invalidResponse: return "家庭服务器回复格式不正确"
        case .invalidInput(let message): return message
        case .server(let code, let detail):
            return Self.identityHints[detail] ?? "家庭服务器（\(code)）：\(detail)"
        case .trust: return "家庭服务器证书未通过配对校验"
        case .network: return "连接家中 Mac 失败。请确认手机连着家中 Wi-Fi，并在系统设置中允许 Puppy 家长访问本地网络"
        }
    }
}

private final class HomeTrust: NSObject, URLSessionDelegate {
    let anchor: SecCertificate
    init?(_ certificate: String) {
        guard let data = Data(base64Encoded: certificate),
              let anchor = SecCertificateCreateWithData(nil, data as CFData) else { return nil }
        self.anchor = anchor
    }
    func urlSession(_ session: URLSession, didReceive challenge: URLAuthenticationChallenge,
                    completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
        guard challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
              let trust = challenge.protectionSpace.serverTrust else {
            completionHandler(.cancelAuthenticationChallenge, nil); return
        }
        SecTrustSetPolicies(trust, SecPolicyCreateSSL(true, challenge.protectionSpace.host as CFString))
        let anchors: [SecCertificate] = [anchor]
        SecTrustSetAnchorCertificates(trust, anchors as CFArray)
        SecTrustSetAnchorCertificatesOnly(trust, true)
        var error: CFError?
        guard SecTrustEvaluateWithError(trust, &error) else {
            completionHandler(.cancelAuthenticationChallenge, nil); return
        }
        completionHandler(.useCredential, URLCredential(trust: trust))
    }
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}

final class MobileAdminClient {
    private let session: URLSession
    private let home: URL
    private let identity: PhoneIdentity
    private let delegate: HomeTrust

    init(pairing: Pairing, identity: PhoneIdentity) throws {
        try pairing.validate()
        guard let address = pairing.adminURL, let home = URL(string: address),
              let certificate = pairing.adminCACertificate, let trust = HomeTrust(certificate) else {
            throw AdminError.unavailable
        }
        self.home = home
        self.identity = identity
        delegate = trust
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 35
        configuration.timeoutIntervalForResource = 40
        configuration.urlCache = nil
        configuration.httpCookieStorage = nil
        session = URLSession(configuration: configuration, delegate: trust, delegateQueue: nil)
    }

    deinit { session.invalidateAndCancel() }

    func request(_ method: String, _ path: String, body: [String: Any]? = nil,
                 timeout: TimeInterval? = nil) async throws -> Data {
        let content = try body.map { try JSONSerialization.data(withJSONObject: $0, options: [.sortedKeys, .fragmentsAllowed]) } ?? Data()
        return try await request(method, path, content: content, contentType: "application/json",
                                 timeout: timeout)
    }

    func uploadVoice(_ wav: Data, for memberID: String) async throws -> [String: Any] {
        guard (96_000...644_096).contains(wav.count) else {
            throw AdminError.invalidInput("请录制 3–20 秒语音")
        }
        let boundary = "puppy-" + UUID().uuidString.replacingOccurrences(of: "-", with: "")
        var content = Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"voice.wav\"\r\nContent-Type: audio/wav\r\n\r\n".utf8)
        content.append(wav)
        content.append(Data("\r\n--\(boundary)--\r\n".utf8))
        let data = try await request("POST", "/api/identity/members/\(memberID)/voice", content: content,
                                     contentType: "multipart/form-data; boundary=\(boundary)")
        guard let value = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw AdminError.invalidResponse
        }
        return value
    }

    func uploadFace(_ jpeg: Data, for memberID: String) async throws -> [String: Any] {
        guard !jpeg.isEmpty, jpeg.count <= 512_000,
              jpeg.starts(with: [0xFF, 0xD8]), jpeg.suffix(2).elementsEqual([0xFF, 0xD9]) else {
            throw AdminError.invalidInput("照片需要是小于 512 KB 的 JPEG，请重新拍摄")
        }
        let boundary = "puppy-" + UUID().uuidString.replacingOccurrences(of: "-", with: "")
        var content = Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"face.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".utf8)
        content.append(jpeg)
        content.append(Data("\r\n--\(boundary)--\r\n".utf8))
        let data = try await request("POST", "/api/identity/members/\(memberID)/face", content: content,
                                     contentType: "multipart/form-data; boundary=\(boundary)")
        guard let value = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw AdminError.invalidResponse
        }
        return value
    }

    private func request(_ method: String, _ path: String, content: Data, contentType: String,
                         timeout: TimeInterval? = nil) async throws -> Data {
        guard ["GET", "POST", "PUT", "PATCH", "DELETE"].contains(method), path.hasPrefix("/api/"),
              path.unicodeScalars.allSatisfy({ $0.value >= 33 && $0.value != 127 }),
              !path.contains("%"), !path.contains("//"), !path.contains(".."),
              !path.contains("?"), !path.contains("#") else { throw AdminError.invalidResponse }
        var target = ""
        for byte in path.utf8 {
            if (65...90).contains(byte) || (97...122).contains(byte) || (48...57).contains(byte)
                || [UInt8(47), 45, 95, 46, 58].contains(byte) {
                target.append(Character(UnicodeScalar(Int(byte))!))
            } else { target += String(format: "%%%02X", byte) }
        }
        guard let url = URL(string: home.absoluteString + target) else { throw AdminError.invalidResponse }
        guard content.count <= 1_048_576 else { throw AdminError.invalidInput("上传内容过大") }
        let timestamp = String(Int(Date().timeIntervalSince1970))
        let nonce = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
        let digest = SHA256.hash(data: content).map { String(format: "%02x", $0) }.joined()
        let statement = "PUPPY-ADMIN-V1\n\(method)\n\(target)\n\(digest)\n\(timestamp)\n\(nonce)"
        let signature = try identity.privateKey.signature(for: Data(statement.utf8)).base64EncodedString()
        var request = URLRequest(url: url)
        if let timeout { request.timeoutInterval = timeout }
        else if path == "/api/status" { request.timeoutInterval = 10 }
        request.httpMethod = method
        request.httpBody = content
        request.setValue(identity.sourceID, forHTTPHeaderField: "X-Puppy-Source")
        request.setValue(timestamp, forHTTPHeaderField: "X-Puppy-Timestamp")
        request.setValue(nonce, forHTTPHeaderField: "X-Puppy-Nonce")
        request.setValue(signature, forHTTPHeaderField: "X-Puppy-Signature")
        request.setValue(contentType, forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        let (data, response): (Data, URLResponse)
        do { (data, response) = try await session.data(for: request) }
        catch let error as URLError where [.timedOut, .cannotFindHost, .cannotConnectToHost,
                                            .networkConnectionLost, .notConnectedToInternet].contains(error.code) {
            throw AdminError.network
        }
        guard data.count <= 2_097_152, let response = response as? HTTPURLResponse else { throw AdminError.invalidResponse }
        if response.statusCode >= 400 {
            let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            let detail = parsed?["detail"] as? String ?? "请求未完成"
            throw AdminError.server(response.statusCode, detail)
        }
        guard (200..<300).contains(response.statusCode) else { throw AdminError.invalidResponse }
        return data
    }

    func json(_ method: String, _ path: String, body: [String: Any]? = nil,
              timeout: TimeInterval? = nil) async throws -> [String: Any] {
        let data = try await request(method, path, body: body, timeout: timeout)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw AdminError.invalidResponse
        }
        return object
    }

    func jsonArray(_ method: String, _ path: String) async throws -> [[String: Any]] {
        let data = try await request(method, path)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw AdminError.invalidResponse
        }
        return object
    }
}
