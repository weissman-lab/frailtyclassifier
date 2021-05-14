[![Build Status](https://travis-ci.com/pennsignals/pennsignals-azure.svg?token=x1KDPzKG3hRqhHm8VAvh&branch=master)](https://travis-ci.com/pennsignals/pennsignals-azure)


## Prerequisites:
 - [Docker](https://www.docker.com/)

## Quick Start

	bash init.sh
	docker exec -it pennsignals_azure_vmss_terraform_1 terraform apply
	or:
	bash init.sh && docker exec -it pennsignals_azure_vmss_terraform_1 terraform apply
	Enter a value: yes
	Login via:
	ssh -i ~/projects/pennsignals_azure_vmss/.ssh/azure/acs_key signals@10.146.0.xx
	Get stuff via: 
	rsync -azvPh -e "ssh -i ~/projects/pennsignals_azure/.ssh/azure/acs_key" signals@10.146.0.xx:LOC LOC


tensorboard cheatsheet:
ssh -i ~/projects/pennsignals_azure_vmss/.ssh/azure/acs_key signals@10.146.0.xx -N -f -L localhost:16006:localhost:6006

Destroy when done:
	
jupyter cheatsheet:
jupyter notebook --no-browser --port=8888
ssh -i ~/projects/pennsignals_azure/.ssh/azure/acs_key -N -f -L localhost:8889:localhost:8888 signals@10.146.0.xx


	docker exec -it pennsignals_azure_vmss_terraform_1 terraform destroy
	Enter a value: yes

## Customize

Modify startup scripts:

	./modules/hello_world/templates/cloud-init
	./modules/hello_world/templates/startup.sh

Modify vmss properties:

	./terraform.tfvars

## Connect

	ssh -i ./.ssh/azure/acs_key signals@ip

# Develop

## Configure

To change vmss VM properties

	./modules/vmss/main.tf

## Get Secrets
```
ssh vault-01

vault read secret/azure/terraform/client_id
vault read secret/azure/terraform/client_secret
vault read secret/azure/terraform/subscription_id
vault read secret/azure/terraform/tenant_id
```
## Prepare environment variable file
To apply Terraform file and build the infrustructure, you will need to add an environment variable file to `environments` directory with following variables as an example:

terraform.tfavars:

	env_ip_c_class = "10.146.0"
	env_name = "hello_world"


Name each environment file to match the env name. i.e.: prod, stagin, jon-dev, etc

## Start Docker container
	docker-compose up -d

## Run Terraform:
Initialize the Terraform working directory and generate an execution plan.
	
	docker exec -it pennsignals_azure_vmss_terraform_1 terraform init
	docker exec -it pennsignals_azure_vmss_terraform_1 terraform plan

Then apply Terrform by passing the `.env` file:

	env $(cat .env) terraform apply -var-file=environments/prod.tfvars


## Increase VM size:
The image sizes and their SKU can be changed in `variables.tf`:

```
variable "nomad_az_vm_size" {
  type        = string
  description = "Specifies the size of Azure virtual machines in a scale set"
  default     = "Standard_A0"
}
```