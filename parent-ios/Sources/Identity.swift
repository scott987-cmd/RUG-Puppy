import Foundation
import CryptoKit
import Security

struct PhoneIdentity {
    let privateKey: Curve25519.Signing.PrivateKey
    var publicKey: String { privateKey.publicKey.rawRepresentation.base64EncodedString() }
    var sourceID: String { "iphone-" + Wire.hash(privateKey.publicKey.rawRepresentation).prefix(24) }

    static func load() throws -> PhoneIdentity {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                   kSecAttrService as String: "org.puppycompanion.parent.signing",
                                   kSecAttrAccount as String: "phone-ed25519-v1",
                                   kSecAttrSynchronizable as String: false]
        var lookup = query
        lookup[kSecReturnData as String] = true
        lookup[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        let result = SecItemCopyMatching(lookup as CFDictionary, &item)
        if result == errSecSuccess, let data = item as? Data {
            return PhoneIdentity(privateKey: try .init(rawRepresentation: data))
        }
        guard result == errSecItemNotFound else { throw ClientError.invalid("无法访问手机钥匙串（\(result)）") }
        let key = Curve25519.Signing.PrivateKey()
        var add = query
        add[kSecValueData as String] = key.rawRepresentation
        add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        let status = SecItemAdd(add as CFDictionary, nil)
        guard status == errSecSuccess else { throw ClientError.invalid("无法保存手机私钥（\(status)）") }
        return PhoneIdentity(privateKey: key)
    }

    func registration(deviceID: String) throws -> String {
        String(decoding: try Wire.encode(["source_id": sourceID, "public_key": publicKey, "device_id": deviceID]), as: UTF8.self)
    }
}

