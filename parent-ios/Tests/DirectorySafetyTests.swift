import XCTest
import CryptoKit
import Darwin
@testable import PuppyParent

final class DirectorySafetyTests: XCTestCase {
    final class Fixture {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathComponent("PuppyCompanion")
        let key = Curve25519.Signing.PrivateKey()
        var pairing: Pairing { Pairing(serverID: "test-server", deviceID: "test-puppy", serverPublicKey: key.publicKey.rawRepresentation.base64EncodedString()) }
        let request: Command
        init(kind: CommandKind = .status) throws {
            let pairing = Pairing(serverID: "test-server", deviceID: "test-puppy", serverPublicKey: key.publicKey.rawRepresentation.base64EncodedString())
            request = try .make(kind: kind, pairing: pairing, sourceID: "test-phone")
            for child in ["receipts", "rejections", "media", "status"] {
                try FileManager.default.createDirectory(at: root.appendingPathComponent(child), withIntermediateDirectories: true)
            }
        }
        deinit { try? FileManager.default.removeItem(at: root.deletingLastPathComponent()) }
        var record: RequestRecord { .init(command: request, publicationConfirmed: true, receipt: nil) }
        func outcome(_ state: ReceiptState = .succeeded, media: MediaReference? = nil, bound: Bool = true) -> Receipt {
            Receipt(version: 1, requestID: request.id, serverID: request.serverID, sourceID: bound ? request.sourceID : nil,
                    deviceID: bound ? request.deviceID : nil, kind: bound ? request.kind.rawValue : nil, state: state,
                    updatedAt: Date().timeIntervalSince1970,
                    result: state == .succeeded ? ReceiptResult(ok: true, status: nil, media: media, capturedAt: nil, error: nil) : nil,
                    error: state == .rejected ? "conflicting_request_id" : nil)
        }
        func write(_ receipt: Receipt, sidecar: Bool = false) throws {
            let path = sidecar ? "rejections/\(request.id)-abcdef0123456789.json" : "receipts/\(request.id).json"
            try Wire.sign(receipt, key: key).write(to: root.appendingPathComponent(path), options: .atomic)
        }
        func read(_ record: RequestRecord) throws -> DirectoryUpdate {
            try DirectoryReader.read(root: root, pairing: pairing, records: [record])
        }
    }

    func testSidecarRejectionFirstCannotBlockCanonicalSuccessAfterRestart() throws {
        let fixture = try Fixture()
        try fixture.write(fixture.outcome(.rejected, bound: false), sidecar: true)
        var record = fixture.record
        let rejectedFile = try fixture.read(record)
        XCTAssertNil(rejectedFile.receipts[record.id])
        XCTAssertEqual(rejectedFile.rejectionNotices[record.id]?.state, .rejected)
        record.rejectionNotice = rejectedFile.rejectionNotices[record.id]
        record = try JSONDecoder().decode(RequestRecord.self, from: Wire.encode(record))
        XCTAssertEqual(record.title(), "已提交")
        try fixture.write(fixture.outcome())
        let canonical = try fixture.read(record)
        XCTAssertEqual(canonical.receipts[record.id]?.state, .succeeded)
        record.receipt = canonical.receipts[record.id]
        XCTAssertEqual(record.title(), "已完成")
    }

    func testCanonicalSuccessFirstRemainsSuccessAfterConflictingSidecar() throws {
        let fixture = try Fixture()
        try fixture.write(fixture.outcome())
        var record = fixture.record
        record.receipt = try fixture.read(record).receipts[record.id]
        try fixture.write(fixture.outcome(.rejected, bound: false), sidecar: true)
        let next = try fixture.read(record)
        XCTAssertNil(next.receipts[record.id])
        XCTAssertNil(next.rejectionNotices[record.id])
        XCTAssertEqual(record.title(), "已完成")
    }

    func testCanonicalRejectionRemainsTerminalAndRequiresCompleteTarget() throws {
        let fixture = try Fixture()
        var record = fixture.record
        try fixture.write(fixture.outcome(.rejected, bound: false))
        XCTAssertNil(try fixture.read(record).receipts[record.id])
        try fixture.write(fixture.outcome(.rejected))
        record.receipt = try fixture.read(record).receipts[record.id]
        XCTAssertEqual(record.receipt?.state, .rejected)
        try fixture.write(fixture.outcome())
        let conflict = try fixture.read(record)
        XCTAssertNil(conflict.receipts[record.id])
        XCTAssertTrue(conflict.notes.contains("收到冲突回执，需核实实际状态"))
    }

    func testLegacyUnboundRejectionMigratesToNotice() throws {
        let fixture = try Fixture()
        var record = fixture.record
        record.receipt = fixture.outcome(.rejected, bound: false)
        try fixture.write(fixture.outcome())
        XCTAssertEqual(try fixture.read(record).receipts[record.id]?.state, .succeeded)
        record.separateLegacyRejection()
        XCTAssertNil(record.receipt)
        XCTAssertEqual(record.rejectionNotice?.state, .rejected)
    }

    func testRegularFileReadsRejectSymlinkLeafSymlinkDirectoryAndFIFO() throws {
        let fixture = try Fixture()
        let outside = fixture.root.deletingLastPathComponent().appendingPathComponent("outside")
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        let secret = outside.appendingPathComponent("secret")
        try Data("must not follow".utf8).write(to: secret)
        let leaf = fixture.root.appendingPathComponent("media/leaf")
        try FileManager.default.createSymbolicLink(at: leaf, withDestinationURL: secret)
        XCTAssertThrowsError(try DirectoryIO.availableData(leaf, maximum: 100, root: fixture.root))
        let linked = fixture.root.appendingPathComponent("linked")
        try FileManager.default.createSymbolicLink(at: linked, withDestinationURL: outside)
        XCTAssertThrowsError(try DirectoryIO.availableData(linked.appendingPathComponent("secret"), maximum: 100, root: fixture.root))
        let pipe = fixture.root.appendingPathComponent("media/pipe")
        XCTAssertEqual(mkfifo(pipe.path, 0o600), 0)
        let start = Date()
        XCTAssertThrowsError(try DirectoryIO.availableData(pipe, maximum: 100, root: fixture.root))
        XCTAssertLessThan(Date().timeIntervalSince(start), 1)
    }

    func testSelectedPairingFileIsBoundedAndRejectsNonregularItems() throws {
        let fixture = try Fixture()
        let file = fixture.root.appendingPathComponent("pairing.json")
        let data = Data("{\"version\":1}".utf8)
        try data.write(to: file)
        XCTAssertEqual(try DirectoryIO.selectedFileData(file, maximum: data.count), data)
        XCTAssertThrowsError(try DirectoryIO.selectedFileData(file, maximum: data.count - 1))
        XCTAssertThrowsError(try DirectoryIO.selectedFileData(fixture.root, maximum: 100))
        let link = fixture.root.appendingPathComponent("pairing-link.json")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: file)
        XCTAssertThrowsError(try DirectoryIO.selectedFileData(link, maximum: 100))
    }

    func testPublishRejectsLinkedCommandsDirectoryAndLinkedRoot() throws {
        let fixture = try Fixture()
        let outside = fixture.root.deletingLastPathComponent().appendingPathComponent("outside")
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: fixture.root.appendingPathComponent("commands"), withDestinationURL: outside)
        let data = try Wire.sign(fixture.request, key: fixture.key)
        XCTAssertThrowsError(try DirectoryIO.publish(data, root: fixture.root, command: fixture.request))
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: outside.path).isEmpty)
        let linkedParent = fixture.root.deletingLastPathComponent().appendingPathComponent("linked-parent")
        try FileManager.default.createDirectory(at: linkedParent, withIntermediateDirectories: true)
        let linkedRoot = linkedParent.appendingPathComponent("PuppyCompanion")
        try FileManager.default.createSymbolicLink(at: linkedRoot, withDestinationURL: outside)
        XCTAssertThrowsError(try DirectoryIO.publish(data, root: linkedRoot, command: fixture.request))
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: outside.path).isEmpty)
    }

    func testExpiredBudgetStartsNoIO() throws {
        let fixture = try Fixture()
        var count = 0
        XCTAssertThrowsError(try DirectoryReader.read(root: fixture.root, pairing: fixture.pairing, records: [fixture.record],
                                                      budget: IOBudget(timeout: 0), readOverride: { _, _ in count += 1; return Data() }))
        XCTAssertEqual(count, 0)
    }

    func testTimedOutReaderDoesNotStartAnotherReadWhenBlockedCallReturns() async throws {
        let fixture = try Fixture()
        let io = DirectoryIO()
        let release = DispatchSemaphore(value: 0)
        let finished = expectation(description: "worker exits after its blocked call returns")
        final class Counter: @unchecked Sendable {
            let lock = NSLock()
            var value = 0
            func increment() { lock.withLock { value += 1 } }
            var count: Int { lock.withLock { value } }
        }
        let counter = Counter()
        do {
            _ = try await io.run(timeout: 1) { budget in
                defer { finished.fulfill() }
                return try DirectoryReader.read(root: fixture.root, pairing: fixture.pairing, records: [fixture.record], budget: budget,
                    readOverride: { _, _ in counter.increment(); release.wait(); return Data("{}".utf8) })
            }
            XCTFail("must time out")
        } catch { XCTAssertEqual(error.localizedDescription, ClientError.timeout.localizedDescription) }
        XCTAssertEqual(counter.count, 1)
        release.signal()
        await fulfillment(of: [finished], timeout: 2)
        XCTAssertEqual(counter.count, 1)
    }

    func testVerifiedMediaHashCacheAvoidsAnotherRead() throws {
        let fixture = try Fixture(kind: .photo)
        let data = Data("verified bytes".utf8)
        let media = MediaReference(file: "media/\(fixture.request.id).jpg", sha256: Wire.hash(data), bytes: data.count)
        try data.write(to: fixture.root.appendingPathComponent(media.file))
        try fixture.write(fixture.outcome(media: media))
        var record = fixture.record
        let first = try fixture.read(record)
        XCTAssertEqual(first.photos[record.id], data)
        record.receipt = first.receipts[record.id]
        var mediaReads = 0
        let cached = try DirectoryReader.read(root: fixture.root, pairing: fixture.pairing, records: [record],
            verifiedMediaHashes: [record.id: media.sha256], readOverride: { url, maximum in
                if url.pathExtension == "jpg" { mediaReads += 1 }
                return try DirectoryIO.availableData(url, maximum: maximum, root: fixture.root)
            })
        XCTAssertEqual(mediaReads, 0)
        XCTAssertTrue(cached.photos.isEmpty)
    }

    func testCachedPhotoBecomesUnavailableAfterMediaRemoval() throws {
        let fixture = try Fixture(kind: .photo)
        let data = Data("verified bytes".utf8)
        let media = MediaReference(file: "media/\(fixture.request.id).jpg", sha256: Wire.hash(data), bytes: data.count)
        let file = fixture.root.appendingPathComponent(media.file)
        try data.write(to: file)
        try fixture.write(fixture.outcome(media: media))
        var record = fixture.record
        record.receipt = try fixture.read(record).receipts[record.id]
        try FileManager.default.removeItem(at: file)
        let update = try DirectoryReader.read(root: fixture.root, pairing: fixture.pairing, records: [record],
                                              verifiedMediaHashes: [record.id: media.sha256])
        XCTAssertTrue(update.unavailableMediaIDs.contains(record.id))
        XCTAssertNil(update.photos[record.id])
    }

    func testCachedPhotoPlaceholderRemainsPending() throws {
        let fixture = try Fixture(kind: .photo)
        let data = Data("verified bytes".utf8)
        let media = MediaReference(file: "media/\(fixture.request.id).jpg", sha256: Wire.hash(data), bytes: data.count)
        let placeholder = fixture.root.appendingPathComponent("media/.\(fixture.request.id).jpg.icloud")
        try Data().write(to: placeholder)
        let receipt = fixture.outcome(media: media)
        try fixture.write(receipt)
        var record = fixture.record
        // This test exercises the cached-photo path. Reading uncached media first
        // would ask File Provider to download the deliberately synthetic placeholder.
        record.receipt = receipt
        let update = try DirectoryReader.read(root: fixture.root, pairing: fixture.pairing, records: [record],
                                              verifiedMediaHashes: [record.id: media.sha256])
        XCTAssertFalse(update.unavailableMediaIDs.contains(record.id))
    }
}
