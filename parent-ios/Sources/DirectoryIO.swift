import Foundation
import Darwin

struct IOBudget: Sendable {
    let deadline: UInt64
    init(timeout: TimeInterval = 8) {
        deadline = DispatchTime.now().uptimeNanoseconds + UInt64(max(0, timeout) * 1_000_000_000)
    }
    func check() throws {
        guard DispatchTime.now().uptimeNanoseconds < deadline else { throw ClientError.timeout }
    }
}

/// A provider call may remain blocked after the UI deadline. One actual worker
/// owns a monotonic budget and must check it before starting each subsequent I/O.
final class DirectoryIO: @unchecked Sendable {
    private let lock = NSLock()
    private var working = false
    private final class Completion<Value>: @unchecked Sendable {
        let lock = NSLock()
        var continuation: CheckedContinuation<Value, Error>?
        init(_ continuation: CheckedContinuation<Value, Error>) { self.continuation = continuation }
        func finish(_ result: Result<Value, Error>) {
            lock.lock()
            let target = continuation
            continuation = nil
            lock.unlock()
            target?.resume(with: result)
        }
    }

    func run<Value>(timeout: TimeInterval = 8, _ operation: @escaping (IOBudget) throws -> Value) async throws -> Value {
        let available = lock.withLock { () -> Bool in
            if working { return false }
            working = true
            return true
        }
        guard available else { throw ClientError.busy }
        let budget = IOBudget(timeout: timeout)
        return try await withCheckedThrowingContinuation { continuation in
            let completion = Completion(continuation)
            DispatchQueue.global(qos: .utility).async {
                let result = Result {
                    try budget.check()
                    let value = try operation(budget)
                    try budget.check()
                    return value
                }
                self.lock.withLock { self.working = false }
                completion.finish(result)
            }
            DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + timeout) {
                completion.finish(.failure(ClientError.timeout))
            }
        }
    }

    static func scoped<Value>(_ url: URL, operation: () throws -> Value) rethrows -> Value {
        let access = url.startAccessingSecurityScopedResource()
        defer { if access { url.stopAccessingSecurityScopedResource() } }
        return try operation()
    }

    /// A document-picker URL grants access to the selected file, not necessarily
    /// its File Provider parent directory. Read only that selected public JSON.
    static func selectedFileData(_ url: URL, maximum: Int, budget: IOBudget = IOBudget()) throws -> Data {
        try scoped(url) {
            try budget.check()
            var coordinatorError: NSError?
            var result: Result<Data, Error>?
            NSFileCoordinator().coordinate(readingItemAt: url, options: [], error: &coordinatorError) { coordinated in
                result = Result {
                    try budget.check()
                    let values = try coordinated.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
                    guard values.isRegularFile == true, values.isSymbolicLink != true else {
                        throw ClientError.invalid("请选择普通 JSON 文件")
                    }
                    let handle = try FileHandle(forReadingFrom: coordinated)
                    defer { try? handle.close() }
                    let data = try handle.read(upToCount: maximum + 1) ?? Data()
                    try budget.check()
                    guard data.count <= maximum else { throw ClientError.invalid("文件超过大小限制") }
                    return data
                }
            }
            if let coordinatorError { throw coordinatorError }
            guard let result else { throw ClientError.invalid("无法读取文件") }
            return try result.get()
        }
    }

    private static func systemError() -> POSIXError { POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
    private static let directoryFlags = O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK

    /// The selected security-scoped root is the authority. Walk every directory
    /// below that root using descriptors; a later pathname swap cannot redirect
    /// the final open/rename. Do not resolve untrusted child symlinks.
    private static func parentDescriptor(_ url: URL, root: URL, budget: IOBudget) throws -> (Int32, String) {
        try budget.check()
        // Foundation may standardize the same /private/var prefix differently
        // for a selected directory and an NSFileCoordinator child on iPadOS.
        // Compare lexical paths; openat below rejects symlinks at every child.
        let anchor = root.pathComponents
        let target = url.pathComponents
        let relative = target.dropFirst(anchor.count)
        guard target.count > anchor.count, Array(target.prefix(anchor.count)) == anchor,
              !relative.contains("."), !relative.contains("..") else {
            throw ClientError.invalid("文件不在已授权目录内")
        }
        var descriptor = open(root.path, directoryFlags)
        guard descriptor >= 0 else { throw systemError() }
        do {
            for component in relative.dropLast() {
                try budget.check()
                let next = openat(descriptor, component, directoryFlags)
                guard next >= 0 else { throw systemError() }
                close(descriptor)
                descriptor = next
            }
            return (descriptor, target.last!)
        } catch { close(descriptor); throw error }
    }

    private static func regular(_ info: stat, maximum: Int) throws {
        guard (info.st_mode & mode_t(S_IFMT)) == mode_t(S_IFREG), info.st_size >= 0,
              info.st_size <= maximum else { throw ClientError.invalid("文件类型或大小不支持") }
    }

    private static func checkLeaf(_ url: URL, root: URL, maximum: Int, budget: IOBudget) throws {
        let (parent, name) = try parentDescriptor(url, root: root, budget: budget)
        defer { close(parent) }
        try budget.check()
        var info = stat()
        guard fstatat(parent, name, &info, AT_SYMLINK_NOFOLLOW) == 0 else { throw systemError() }
        try regular(info, maximum: maximum)
    }

    static func availableData(_ url: URL, maximum: Int, root: URL? = nil, budget: IOBudget = IOBudget()) throws -> Data {
        let anchor = root ?? url.deletingLastPathComponent()
        let manager = FileManager.default
        let placeholder = url.deletingLastPathComponent().appendingPathComponent("." + url.lastPathComponent + ".icloud")
        do { try checkLeaf(url, root: anchor, maximum: maximum, budget: budget) }
        catch let error as POSIXError where error.code == .ENOENT {
            try budget.check()
            if (try? checkLeaf(placeholder, root: anchor, maximum: maximum, budget: budget)) != nil {
                try budget.check()
                try manager.startDownloadingUbiquitousItem(at: placeholder)
                throw ClientError.downloading
            }
            throw error
        }
        try budget.check()
        let values = try url.resourceValues(forKeys: [.ubiquitousItemDownloadingStatusKey])
        if let status = values.ubiquitousItemDownloadingStatus, status != .current {
            try budget.check()
            try manager.startDownloadingUbiquitousItem(at: url)
            throw ClientError.downloading
        }
        try budget.check()
        var coordinatorError: NSError?
        var result: Result<Data, Error>?
        NSFileCoordinator().coordinate(readingItemAt: url, options: [], error: &coordinatorError) { coordinated in
            result = Result {
                let (parent, name) = try parentDescriptor(coordinated, root: anchor, budget: budget)
                defer { close(parent) }
                try budget.check()
                let descriptor = openat(parent, name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
                guard descriptor >= 0 else { throw systemError() }
                let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
                defer { try? handle.close() }
                try budget.check()
                var info = stat()
                guard fstat(descriptor, &info) == 0 else { throw systemError() }
                try regular(info, maximum: maximum)
                try budget.check()
                let data = try handle.read(upToCount: maximum + 1) ?? Data()
                try budget.check()
                guard data.count <= maximum else { throw ClientError.invalid("文件超过大小限制") }
                return data
            }
        }
        if let coordinatorError { throw coordinatorError }
        guard let result else { throw ClientError.invalid("无法读取文件") }
        return try result.get()
    }

    /// Check a previously verified photo without rereading its bytes every refresh.
    /// A File Provider placeholder is pending, while absent local and placeholder
    /// names mean the photo can no longer be displayed from this directory.
    static func mediaPresentOrPending(_ url: URL, root: URL, budget: IOBudget) throws -> Bool {
        do {
            try checkLeaf(url, root: root, maximum: 512_000, budget: budget)
            return true
        } catch let error as POSIXError where error.code == .ENOENT {
            try budget.check()
            let placeholder = url.deletingLastPathComponent().appendingPathComponent("." + url.lastPathComponent + ".icloud")
            do {
                try checkLeaf(placeholder, root: root, maximum: 512_000, budget: budget)
                return true
            } catch let placeholderError as POSIXError where placeholderError.code == .ENOENT {
                return false
            }
        }
    }

    static func publish(_ data: Data, root: URL, command: Command, budget: IOBudget = IOBudget()) throws {
        try scoped(root) {
            guard root.lastPathComponent == "PuppyCompanion", UUID(uuidString: command.requestID) != nil,
                  data.count <= 131_072 else { throw ClientError.invalid("目录或请求无效") }
            func check() throws {
                try budget.check()
                guard Date().timeIntervalSince1970 < Double(command.expiresAt) else { throw ClientError.invalid("请求已过期，未提交") }
            }
            try check()
            let rootFD = open(root.path, directoryFlags)
            guard rootFD >= 0 else { throw systemError() }
            defer { close(rootFD) }
            try check()
            guard mkdirat(rootFD, "commands", 0o700) == 0 || errno == EEXIST else { throw systemError() }
            let url = root.appendingPathComponent("commands/\(command.requestID).json")
            // Validate the parent before asking File Provider to coordinate it.
            let (preflight, _) = try parentDescriptor(url, root: root, budget: budget)
            close(preflight)
            try check()
            var coordinatorError: NSError?
            var result: Result<Void, Error>?
            NSFileCoordinator().coordinate(writingItemAt: url, options: [], error: &coordinatorError) { destination in
                result = Result {
                    let (parent, name) = try parentDescriptor(destination, root: root, budget: budget)
                    defer { close(parent) }
                    try check()
                    let temporary = ".\(UUID().uuidString).tmp"
                    let descriptor = openat(parent, temporary, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
                    guard descriptor >= 0 else { throw systemError() }
                    defer { close(descriptor); unlinkat(parent, temporary, 0) }
                    try data.withUnsafeBytes { buffer in
                        var offset = 0
                        while offset < data.count {
                            try check()
                            let count = Darwin.write(descriptor, buffer.baseAddress!.advanced(by: offset), data.count - offset)
                            if count < 0 && errno == EINTR { continue }
                            guard count > 0 else { throw systemError() }
                            offset += count
                        }
                    }
                    try check()
                    guard fsync(descriptor) == 0 else { throw systemError() }
                    try check()
                    // Atomic, no replacement, and anchored to verified directory
                    // descriptors on both sides (iOS 10+).
                    guard renameatx_np(parent, temporary, parent, name, UInt32(RENAME_EXCL)) == 0 else { throw systemError() }
                }
            }
            if let coordinatorError { throw coordinatorError }
            guard let result else { throw ClientError.invalid("无法提交文件") }
            try result.get()
        }
    }

    static func rejectionFiles(root: URL, budget: IOBudget, limit: Int = 256) throws -> [String: [URL]] {
        let folder = root.appendingPathComponent("rejections", isDirectory: true)
        let (parent, name) = try parentDescriptor(folder, root: root, budget: budget)
        defer { close(parent) }
        try budget.check()
        let descriptor = openat(parent, name, directoryFlags)
        guard descriptor >= 0 else { throw systemError() }
        do { try budget.check() } catch { close(descriptor); throw error }
        guard let stream = fdopendir(descriptor) else { close(descriptor); throw systemError() }
        defer { closedir(stream) }
        var files: [String: [URL]] = [:]
        for _ in 0..<limit {
            try budget.check()
            guard let entry = readdir(stream) else { break }
            let name = withUnsafePointer(to: &entry.pointee.d_name) {
                $0.withMemoryRebound(to: CChar.self, capacity: Int(entry.pointee.d_namlen) + 1) { String(cString: $0) }
            }
            guard name.count > 37, name.hasSuffix(".json") else { continue }
            let id = String(name.prefix(36))
            guard UUID(uuidString: id) != nil, name.dropFirst(36).first == "-" else { continue }
            files[id, default: []].append(folder.appendingPathComponent(name))
        }
        return files
    }
}

struct DirectoryUpdate {
    var snapshot: Snapshot?
    var receipts: [String: Receipt] = [:]
    var rejectionNotices: [String: Receipt] = [:]
    var photos: [String: Data] = [:]
    var unavailableMediaIDs: Set<String> = []
    var notes: [String] = []
}

enum DirectoryReader {
    static func read(root: URL, pairing: Pairing, records: [RequestRecord], budget: IOBudget = IOBudget(),
                     verifiedMediaHashes: [String: String] = [:],
                     readOverride: ((URL, Int) throws -> Data)? = nil) throws -> DirectoryUpdate {
        try budget.check()
        return try DirectoryIO.scoped(root) {
            var update = DirectoryUpdate()
            func load(_ url: URL, _ maximum: Int) throws -> Data {
                try budget.check()
                if let readOverride { return try readOverride(url, maximum) }
                return try DirectoryIO.availableData(url, maximum: maximum, root: root, budget: budget)
            }
            let state = root.appendingPathComponent("status/\(pairing.deviceID).json")
            do {
                let value = try Wire.verified(load(state, 131_072), publicKey: pairing.serverPublicKey, as: Snapshot.self)
                try value.validate(pairing: pairing)
                update.snapshot = value
            } catch { update.notes.append(error.localizedDescription) }
            try budget.check()
            // A single bounded directory pass for diagnostics, never per record.
            let rejections = (try? DirectoryIO.rejectionFiles(root: root, budget: budget)) ?? [:]
            for original in records.prefix(100) {
                try budget.check()
                var record = original
                record.separateLegacyRejection()
                let path = root.appendingPathComponent("receipts/\(record.id).json")
                var receipt = record.receipt
                do {
                    let incoming = try Wire.verified(load(path, 131_072), publicKey: pairing.serverPublicKey, as: Receipt.self)
                    try incoming.validate(command: record.command, requireFullTarget: true)
                    if try incoming.supersedes(record.receipt) {
                        receipt = incoming
                        update.receipts[record.id] = incoming
                    }
                } catch {
                    if (error as? POSIXError)?.code != .ENOENT { update.notes.append(error.localizedDescription) }
                }
                // Sidecar rejections can refer to a different command with the
                // same UUID. They are diagnostics, never authoritative outcomes.
                if receipt == nil {
                    var notice = record.rejectionNotice
                    for file in rejections[record.id] ?? [] {
                        try budget.check()
                        do {
                            let incoming = try Wire.verified(load(file, 131_072), publicKey: pairing.serverPublicKey, as: Receipt.self)
                            try incoming.validate(command: record.command)
                            guard incoming.state == .rejected, incoming.updatedAt >= (notice?.updatedAt ?? 0) else { continue }
                            notice = incoming
                            update.rejectionNotices[record.id] = incoming
                        } catch { update.notes.append(error.localizedDescription) }
                    }
                }
                if receipt?.state == .succeeded, let media = receipt?.result?.media {
                    try budget.check()
                    guard media.file == "media/\(record.id).jpg" else {
                        update.unavailableMediaIDs.insert(record.id)
                        update.notes.append("照片路径不匹配")
                        continue
                    }
                    let mediaURL = root.appendingPathComponent(media.file)
                    if verifiedMediaHashes[record.id] == media.sha256, readOverride == nil {
                        do {
                            if !((try DirectoryIO.mediaPresentOrPending(mediaURL, root: root, budget: budget))) {
                                update.unavailableMediaIDs.insert(record.id)
                            }
                        } catch {
                            update.unavailableMediaIDs.insert(record.id)
                            update.notes.append(error.localizedDescription)
                        }
                        continue
                    }
                    if verifiedMediaHashes[record.id] == media.sha256 { continue }
                    do {
                        let data = try load(mediaURL, 512_000)
                        try media.validate(data, requestID: record.id)
                        update.photos[record.id] = data
                    } catch {
                        update.unavailableMediaIDs.insert(record.id)
                        update.notes.append(error.localizedDescription)
                    }
                }
            }
            try budget.check()
            return update
        }
    }
}
