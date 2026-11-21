"""Private TLS verification on loopback only; no system trust or device changes."""
import asyncio
import datetime
import ipaddress
from pathlib import Path
import ssl
import tempfile
import unittest

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.x509.oid import ExtendedKeyUsageOID

from .tls import create_local_certificates, tls_options


class TlsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="puppy-tls-test-")
        self.root = Path(self.temp.name)
        self.paths = create_local_certificates(self.root / "tls", ["localhost", "127.0.0.1"])
        self.config = {"PUPPY_TLS_CERT_PATH": "tls/server.pem", "PUPPY_TLS_KEY_PATH": "tls/server-key.pem"}
        self.original_chain = (self.root / "tls/server.pem").read_bytes()

    async def asyncTearDown(self):
        self.temp.cleanup()

    def replace_certificate(self, position=0, *, start=None, end=None, eku="keep"):
        """Re-sign one test certificate; never read host certificate material."""
        chain = x509.load_pem_x509_certificates(self.original_chain)
        original = chain[position]
        signer = serialization.load_pem_private_key((self.root / "tls/ca-key.pem").read_bytes(), password=None)
        builder = (x509.CertificateBuilder().subject_name(original.subject).issuer_name(original.issuer)
                   .public_key(original.public_key()).serial_number(original.serial_number)
                   .not_valid_before(start or original.not_valid_before_utc)
                   .not_valid_after(end or original.not_valid_after_utc))
        for extension in original.extensions:
            if isinstance(extension.value, x509.ExtendedKeyUsage) and eku != "keep":
                continue
            builder = builder.add_extension(extension.value, extension.critical)
        if eku != "keep" and eku is not None:
            builder = builder.add_extension(x509.ExtendedKeyUsage(eku), critical=False)
        chain[position] = builder.sign(signer, hashes.SHA256())
        (self.root / "tls/server.pem").write_bytes(b"".join(
            cert.public_bytes(serialization.Encoding.PEM) for cert in chain))

    async def test_valid_chain_and_hostname_are_required_on_loopback(self):
        options = tls_options(self.config, self.root)
        context = ssl.SSLContext(options["ssl_version"])
        context.load_cert_chain(options["ssl_certfile"], options["ssl_keyfile"])
        async def respond(reader, writer):
            writer.write(b"verified\n")
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        server = await asyncio.start_server(respond, "127.0.0.1", 0, ssl=context)
        try:
            port = server.sockets[0].getsockname()[1]
            trusted = ssl.create_default_context(cafile=str(self.root / "tls/ca.pem"))
            trusted.verify_flags |= ssl.VERIFY_X509_STRICT
            reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl=trusted, server_hostname="localhost")
            self.assertEqual(await reader.readline(), b"verified\n")
            self.assertIn(writer.get_extra_info("ssl_object").version(), {"TLSv1.2", "TLSv1.3"})
            writer.close()
            await writer.wait_closed()
            for verify_context, hostname in ((trusted, "different.local"), (ssl.create_default_context(), "localhost")):
                with self.assertRaises(ssl.SSLCertVerificationError):
                    await asyncio.open_connection("127.0.0.1", port, ssl=verify_context, server_hostname=hostname)
        finally:
            server.close()
            await server.wait_closed()

    async def test_missing_half_mismatched_keys_and_permissions_fail_closed(self):
        other = create_local_certificates(self.root / "other", ["localhost"])
        cases = ({"PUPPY_TLS_CERT_PATH": self.config["PUPPY_TLS_CERT_PATH"]},
                 {"PUPPY_TLS_KEY_PATH": self.config["PUPPY_TLS_KEY_PATH"]},
                 {**self.config, "PUPPY_TLS_KEY_PATH": other["key_path"]})
        for config in cases:
            with self.assertRaises((ValueError, OSError)):
                tls_options(config, self.root)
        (self.root / "tls/server-key.pem").chmod(0o644)
        with self.assertRaises(ValueError):
            tls_options(self.config, self.root)

    async def test_only_absent_tls_fields_allow_http(self):
        self.assertEqual(tls_options({}, self.root), {})
        self.assertEqual(tls_options({"PUPPY_PORT": 8000}, self.root), {})
        for field in self.config:
            for value in (None, "", "  ", False, 0, [], {}):
                for config in ({field: value}, {**self.config, field: value}):
                    with self.subTest(config=config), self.assertRaises(ValueError):
                        tls_options(config, self.root)
        with self.assertRaises(ValueError):
            tls_options(dict.fromkeys(self.config), self.root)

    async def test_expired_or_future_leaf_is_rejected(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        for first_day, last_day in ((-2, -1), (1, 2)):
            self.replace_certificate(start=now + datetime.timedelta(days=first_day),
                                     end=now + datetime.timedelta(days=last_day))
            with self.subTest(first_day=first_day), self.assertRaisesRegex(ValueError, "证书链"):
                tls_options(self.config, self.root)

    async def test_expired_or_future_issuer_is_rejected(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        for first_day, last_day in ((-2, -1), (1, 2)):
            self.replace_certificate(1, start=now + datetime.timedelta(days=first_day),
                                     end=now + datetime.timedelta(days=last_day))
            with self.subTest(first_day=first_day), self.assertRaisesRegex(ValueError, "证书链"):
                tls_options(self.config, self.root)

    async def test_non_server_leaf_eku_is_rejected(self):
        for usage in (ExtendedKeyUsageOID.CLIENT_AUTH, ExtendedKeyUsageOID.CODE_SIGNING,
                      ExtendedKeyUsageOID.ANY_EXTENDED_KEY_USAGE):
            self.replace_certificate(eku=[usage])
            with self.subTest(usage=usage), self.assertRaisesRegex(ValueError, "服务端"):
                tls_options(self.config, self.root)

    async def test_absent_eku_or_explicit_server_usage_is_allowed(self):
        for usage in (None, [ExtendedKeyUsageOID.SERVER_AUTH],
                      [ExtendedKeyUsageOID.CLIENT_AUTH, ExtendedKeyUsageOID.SERVER_AUTH]):
            self.replace_certificate(eku=usage)
            with self.subTest(usage=usage):
                options = tls_options(self.config, self.root)
                self.assertEqual(options["ssl_certfile"], str((self.root / "tls/server.pem").resolve()))

    async def test_missing_file_directory_and_malformed_chain_are_rejected(self):
        for field in self.config:
            for value in ("missing.pem", "tls"):
                with self.subTest(field=field, value=value), self.assertRaises((ValueError, OSError)):
                    tls_options({**self.config, field: value}, self.root)
        for invalid in (b"not a certificate", self.original_chain +
                        b"-----BEGIN CERTIFICATE-----\ninvalid\n-----END CERTIFICATE-----\n"):
            (self.root / "tls/server.pem").write_bytes(invalid)
            with self.assertRaises(ValueError):
                tls_options(self.config, self.root)

    async def test_creation_preserves_existing_keys_and_keeps_files_private(self):
        before = (self.root / "tls/server-key.pem").read_bytes()
        with self.assertRaises(ValueError):
            create_local_certificates(self.root / "tls", ["localhost"])
        self.assertEqual((self.root / "tls/server-key.pem").read_bytes(), before)
        self.assertEqual((self.root / "tls").stat().st_mode & 0o777, 0o700)
        self.assertTrue(all(path.stat().st_mode & 0o777 == 0o600 for path in (self.root / "tls").iterdir()))
        self.assertFalse(self.paths["system_trust_changed"])
        self.assertFalse(self.paths["server_configuration_changed"])

    async def test_generated_chain_has_expected_ca_server_and_host_constraints(self):
        leaf, ca = x509.load_pem_x509_certificates(self.original_chain)
        leaf.verify_directly_issued_by(ca)
        ca.verify_directly_issued_by(ca)
        self.assertEqual(ca.extensions.get_extension_for_class(x509.BasicConstraints).value,
                         x509.BasicConstraints(ca=True, path_length=0))
        self.assertFalse(leaf.extensions.get_extension_for_class(x509.BasicConstraints).value.ca)
        self.assertEqual(list(leaf.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value),
                         [ExtendedKeyUsageOID.SERVER_AUTH])
        names = leaf.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        self.assertEqual(names.get_values_for_type(x509.DNSName), ["localhost"])
        self.assertEqual(names.get_values_for_type(x509.IPAddress), [ipaddress.ip_address("127.0.0.1")])
        self.assertLessEqual(ca.not_valid_before_utc, leaf.not_valid_before_utc)
        self.assertLessEqual(leaf.not_valid_after_utc, ca.not_valid_after_utc)

    async def test_invalid_generation_hosts_do_not_create_files(self):
        output = self.root / "invalid-host-output"
        cases = (None, "localhost", [], [127], [None], [["localhost"]], [""], [" localhost"],
                 ["a..local"], ["bad_host"], ["256.1.2.3"], ["01.2.3.4"], ["测试.local"],
                 [f"host{index}.local" for index in range(17)])
        for hosts in cases:
            with self.subTest(hosts=hosts), self.assertRaises(ValueError):
                create_local_certificates(output, hosts)
            self.assertFalse(output.exists())

    async def test_generation_deduplicates_hosts(self):
        result = create_local_certificates(self.root / "deduplicated", ["localhost", "localhost"])
        self.assertEqual(result["hosts"], ["localhost"])


if __name__ == "__main__":
    unittest.main()
