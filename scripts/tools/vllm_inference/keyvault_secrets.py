import argparse

from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient


def get_client(kv_uri, client_id):
    credential = ManagedIdentityCredential(client_id=client_id)
    client = SecretClient(vault_url=kv_uri, credential=credential)
    return client


def set_secret(client, secret_name, secret_value):
    client.set_secret(secret_name, secret_value)


def get_secret(client, secret_name, output_only_secret):
    secret = client.get_secret(secret_name)
    if output_only_secret:
        return secret.value
    else:
        return secret.id, secret.name, secret.value


def delete_secret(client, secret_name):
    poller = client.begin_delete_secret(secret_name)
    deleted_secret = poller.result()
    return deleted_secret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-uri", type=str, default="https://aifrontiers.vault.azure.net/")
    parser.add_argument("--client-id", type=str, default="b32444ac-27e2-4f36-ab71-b664f6876f00")
    parser.add_argument("--secret-name", type=str, required=True)
    parser.add_argument("--secret-value", type=str, default=None)
    parser.add_argument("--get", action="store_true")
    parser.add_argument("--set", action="store_true")
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--output_only_secret", action="store_true")
    args = parser.parse_args()

    client = get_client(args.kv_uri, args.client_id)

    if args.get:
        print(get_secret(client, args.secret_name, args.output_only_secret))
    elif args.set:
        if not args.secret_value:
            raise ValueError("Secret value is required for --set.")
        set_secret(client, args.secret_name, args.secret_value)
        print(f"Secret '{args.secret_name}' set successfully.")
    elif args.delete:
        delete_secret(client, args.secret_name)
        print(f"Secret '{args.secret_name}' deleted successfully.")
    else:
        print("No action specified. Use --get, --set, or --delete.")
