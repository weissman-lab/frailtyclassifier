#!/bin/bash
#init.sh

# Get secrets from on-prem vault (must be on VPN)
ssh vault-01 "(\
vault read secret/azure/terraform/client_id | grep value | sed -e 's/ //g' |  sed -e 's/value/TF_VAR_client_id=/'; \
vault read secret/azure/terraform/client_secret | grep value | sed -e 's/ //g' |  sed -e 's/value/TF_VAR_client_secret=/'; \
vault read secret/azure/terraform/subscription_id | grep value | sed -e 's/ //g' |  sed -e 's/value/TF_VAR_subscription_id=/'; \
vault read secret/azure/terraform/tenant_id | grep value | sed -e 's/ //g' |  sed -e 's/value/TF_VAR_tenant_id=/'; \
)" > secrets.env

docker-compose up -d

# Copy ssh keys generated to local .ssh folder
docker cp pennsignals_azure_vmss_terraform_1:/root/.ssh .


docker exec -it pennsignals_azure_vmss_terraform_1 terraform init
docker exec -it pennsignals_azure_vmss_terraform_1 terraform plan