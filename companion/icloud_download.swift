// Only requests File Provider downloads. It never reads command contents,
// changes system permissions, deletes files, or considers a request completed.
import Foundation

struct Input: Decodable {
    let files: [String]
}

struct Item: Encodable {
    let index: Int
    let state: String
    let errorDomain: String?
    let errorCode: Int?
}

func inspect(_ path: String, index: Int) -> Item {
    let manager = FileManager.default
    let original = URL(fileURLWithPath: path)
    let placeholder = original.lastPathComponent.hasPrefix(".") && original.pathExtension == "icloud"
    let url: URL
    if placeholder {
        let name = String(original.deletingPathExtension().lastPathComponent.dropFirst())
        url = original.deletingLastPathComponent().appendingPathComponent(name)
    } else {
        url = original
    }
    do {
        // A previously downloaded ordinary .json can also be outdated. Only
        // `.current` proves the latest known iCloud version is local.
        let values = try url.resourceValues(forKeys: [
            .isUbiquitousItemKey, .ubiquitousItemDownloadingStatusKey,
            .isSymbolicLinkKey, .isRegularFileKey
        ])
        if values.isSymbolicLink == true {
            return Item(index: index, state: "rejected", errorDomain: nil, errorCode: nil)
        }
        if values.isUbiquitousItem != true && !placeholder {
            return Item(index: index, state: "local", errorDomain: nil, errorCode: nil)
        }
        if values.ubiquitousItemDownloadingStatus == .current && !placeholder {
            return Item(index: index, state: "current", errorDomain: nil, errorCode: nil)
        }
        try manager.startDownloadingUbiquitousItem(at: url)
        return Item(index: index, state: "requested", errorDomain: nil, errorCode: nil)
    } catch {
        // Old-style .icloud placeholders may not have an original URL yet.
        // Foundation accepts a download request for the placeholder itself.
        if placeholder {
            do {
                try manager.startDownloadingUbiquitousItem(at: original)
                return Item(index: index, state: "requested", errorDomain: nil, errorCode: nil)
            } catch {
                let failure = error as NSError
                return Item(index: index, state: "failed", errorDomain: failure.domain, errorCode: failure.code)
            }
        }
        let failure = error as NSError
        return Item(index: index, state: "failed", errorDomain: failure.domain, errorCode: failure.code)
    }
}

do {
    let data = FileHandle.standardInput.readDataToEndOfFile()
    guard data.count <= 300_000 else { throw NSError(domain: "PuppyDownload", code: 1) }
    let input = try JSONDecoder().decode(Input.self, from: data)
    guard input.files.count <= 256 else { throw NSError(domain: "PuppyDownload", code: 2) }
    let output = input.files.enumerated().map { inspect($0.element, index: $0.offset) }
    FileHandle.standardOutput.write(try JSONEncoder().encode(output))
} catch {
    // Do not print potentially private paths or JSON from decoding errors.
    FileHandle.standardError.write(Data("download_helper_invalid_input\n".utf8))
    exit(1)
}
