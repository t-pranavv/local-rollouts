# python register_environment.py --environment_name reasoning-lm-tooluse --version "220825" --image_name aifrontiers.azurecr.io/aif/reasoning-lm-tooluse:220825 --managed_identity False
# python register_environment.py --subscription_id 5c9e4789-4852-4ffe-8551-d682affcbd74 --resource_group ai-frontiers-rg --workspace_name ai-frontiers-sa-ws --environment_name reasoning-lm-tooluse --version "220825" --image_name aifrontierssacr.azurecr.io/aif/reasoning-lm-tooluse:220825 --managed_identity False
from fire import Fire

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment

from azure.identity import ManagedIdentityCredential, AzureCliCredential


def azure_auth(
    managed_identity=True,
    user_managed=True,
    client_id="b32444ac-27e2-4f36-ab71-b664f6876f00",
):
    if managed_identity:
        if user_managed and client_id:
            # Use user-assigned managed identity with provided client_id
            credential = ManagedIdentityCredential(client_id=client_id)
        else:
            # Use system-assigned managed identity
            credential = ManagedIdentityCredential()
    else:
        # Fall back to AzureCliCredential if not using managed identity
        credential = AzureCliCredential()
    return credential


def publish_image_to_environment(
    subscription_id="d4fe558f-6660-4fe7-99ec-ae4716b5e03f",
    resource_group="aifrontiers",
    workspace_name="aifrontiers_ws",
    managed_identity=True,
    user_managed=True,
    client_id="b32444ac-27e2-4f36-ab71-b664f6876f00",
    environment_name="",
    version="",
    description="",
    image_name="",
    build_context="",
    conda_file=None,
    datastore=None,
):
    ml_client = MLClient(
        credential=azure_auth(
            managed_identity=managed_identity,
            user_managed=user_managed,
            client_id=client_id,
        ),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    if not environment_name:
        raise ValueError("Environment name is required.")
    if not version:
        raise ValueError("Version name is required.")
    else:
        version = str(version)
    if not image_name and not build_context:
        raise ValueError("Image name (ACR image URI) or build_context is required.")

    if image_name:
        environment = Environment(
            name=environment_name,
            version=version,
            description=description,
            image=image_name,
            datastore=datastore,
        )
    else:
        environment = Environment(
            name=environment_name,
            version=version,
            description=description,
            build=build_context,
            conda_file=conda_file,
            datastore=datastore,
        )

    try:
        ml_client.environments.create_or_update(environment)
        print(f"Environment '{environment_name}:{version}' created successfully using image '{image_name}' in ACR.")
    except Exception as e:
        print(f"Failed to publish environment: {e}")


if __name__ == "__main__":
    Fire(publish_image_to_environment)
