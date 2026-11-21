"""Local TLS material and strict opt-in startup configuration.

Creating certificates does not trust them system-wide or alter a running service.
The home CA must be provisioned to clients before switching Puppy to wss.
"""
import argparse
import datetime
import hashlib
import ipaddress
import os
from pathlib import Path
import re
import ssl


def tls_options(config, root):
    fields = ("PUPPY_TLS_CERT_PATH", "PUPPY_TLS_KEY_PATH")
    if not any(field in config for field in fields):
        return {}
    certificate, key = (config.get(field) for field in fields)
    if not all(isinstance(value, str) and value.strip() for value in (certificate, key)):
        raise ValueError("TLS 证书和私钥路径必须一起配置")
    files = []
    for value in (certificate, key):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(root) / path
        path = path.resolve(strict=True)
        if not path.is_file():
            raise ValueError("TLS 文件不可用")
        files.append(path)
    if files[1].stat().st_mode & 0o077:
        raise ValueError("TLS 私钥仅能由当前用户读取（权限 600）")
    from cryptography import x509
    from cryptography.x509.oid import ExtendedKeyUsageOID

    # OpenSSL's load_cert_chain checks key compatibility, not certificate dates
    # or server usage. Check every supplied certificate before starting TLS.
    certificates = x509.load_pem_x509_certificates(files[0].read_bytes())
    now = datetime.datetime.now(datetime.timezone.utc)
    if any(not (cert.not_valid_before_utc <= now <= cert.not_valid_after_utc)
           for cert in certificates):
        raise ValueError("TLS 证书链包含尚未生效或已过期的证书")
    try:
        usage = certificates[0].extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    except x509.ExtensionNotFound:
        pass  # A certificate without EKU does not restrict its extended usage.
    else:
        if ExtendedKeyUsageOID.SERVER_AUTH not in usage:
            raise ValueError("TLS 叶证书不允许用于服务端身份验证")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(str(files[0]), str(files[1]))
    return {"ssl_certfile": str(files[0]), "ssl_keyfile": str(files[1]),
            "ssl_version": ssl.PROTOCOL_TLS_SERVER}


def create_local_certificates(output, hosts):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    if not isinstance(hosts, (list, tuple)) or not all(isinstance(host, str) for host in hosts):
        raise ValueError("TLS 主机名必须是文字列表")
    hosts = list(dict.fromkeys(hosts))
    if not 1 <= len(hosts) <= 16:
        raise ValueError("请明确指定 1 至 16 个主机名或 IP")
    names = []
    for host in hosts:
        try:
            names.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            if (len(host) > 253 or not host.isascii() or re.fullmatch(r"[0-9.]+", host)
                    or not all(re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?", part)
                               for part in host.split("."))):
                raise ValueError("无效 TLS 主机名")
            names.append(x509.DNSName(host))
    directory = Path(output).expanduser().absolute()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("证书目录必须为空；不会覆盖已有密钥")
    directory.chmod(0o700)
    now = datetime.datetime.now(datetime.timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Puppy Home Private CA")])
    ca = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name).public_key(ca_key.public_key())
          .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(minutes=5))
          .not_valid_after(now + datetime.timedelta(days=3650))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
          .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False, key_cert_sign=True, crl_sign=True,
                encipher_only=False, decipher_only=False), critical=True)
          .sign(ca_key, hashes.SHA256()))
    server_key = ec.generate_private_key(ec.SECP256R1())
    leaf = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Puppy Home Gateway")]))
            .issuer_name(ca_name).public_key(server_key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5)).not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(names), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(server_key.public_key()), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False, key_cert_sign=False, crl_sign=False,
                encipher_only=False, decipher_only=False), critical=True)
            .sign(ca_key, hashes.SHA256()))
    files = {"ca.pem": ca.public_bytes(serialization.Encoding.PEM),
             "server.pem": leaf.public_bytes(serialization.Encoding.PEM) + ca.public_bytes(serialization.Encoding.PEM),
             "ca-key.pem": ca_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                serialization.NoEncryption()),
             "server-key.pem": server_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                        serialization.NoEncryption())}
    for name, data in files.items():
        path = directory / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
    return {"directory": str(directory), "hosts": hosts,
            "ca_sha256": hashlib.sha256(ca.public_bytes(serialization.Encoding.DER)).hexdigest(),
            "certificate_path": str(directory / "server.pem"), "key_path": str(directory / "server-key.pem"),
            "system_trust_changed": False, "server_configuration_changed": False}


def main():
    import json
    parser = argparse.ArgumentParser(description="创建 Puppy 家庭 TLS 文件，不安装系统信任、不启用服务")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(create_local_certificates(args.output, args.host), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
