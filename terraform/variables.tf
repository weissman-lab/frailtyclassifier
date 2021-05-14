# Azure provider information
variable "client_id" {
  type        = string
  description = "Azure client id"
}

variable "client_secret" {
  type        = string
  description = "Azure client secret"
}

variable "subscription_id" {
  type        = string
  description = "Azure subscription id"
}

variable "tenant_id" {
  type        = string
  description = "Azure tenant id"
}

# Environment globals
variable "env_name" {
  type        = string
  description = "Signals environment name"
}

variable "env_ip_c_class" {
  type        = string
  description = "The first three octets for the environment. Used to calulate subnets for different layers"
}


variable "admin_name" {
  type        = string
  description = "Azure VM instance admin account name"
  default     = "signals"
}

variable "admin_ssh_pub_key_file" {
  type        = string
  description = "Admin account SSH public key file path"
}


variable "tags" {
  type        = map(string)
  description = "A mapping of tags to assign to the resource"
  default     = {}
}


# Consul vars
variable "consul_version" {
  type        = string
  description = "Version of Consul"
  default     = "1.4.0"
}

variable "consul_template_version" {
  type        = string
  description = "consul-template version to use"
  default     = "0.19.5"
}

# Quay
variable "quay_auth" {
  description = "Username and Password for quay.io image repository"
}

variable "consul_root_ca" {
  description = "path to Consul-generated root certificate authority for Consul TLS"
}

variable "consul_root_ca_key" {
  description = "path to Consul-generated root certificate authority private key for Consul TLS"
}

variable "hello_world_version" {
  type        = string
  description = "hello_world version"
  default     = "0.8.6"
}

variable "hello_world_capacity" {
  type        = string
  description = "Provides the number of VMs for hello_world cluster"
  default     = "3"
}

variable "hello_world_address" {
  type        = string
  description = "hello_world address"
  default     = "hello_world.consul.services:8080"
}

variable "hello_world_az_vm_size" {
  type        = string
  description = "Specifies the size of Azure virtual machines in a scale set"
  default     = "Standard_A0"
}

variable "hello_world_az_vm_tier" {
  type        = string
  description = "Specifies the Azure tier of virtual machines in a scale set"
  default     = "Standard"
}

variable "hello_world_az_managed_disk_type" {
  type        = string
  description = "Specifies the type of Azure managed disk to create"
  default     = "Standard_LRS"
}

variable "hello_world_az_disk_size_gb" {
  type        = string
  description = "Specifies the size of the Azure managed disk to create in gigabytes"
  default     = "10"
}

# ACD added Jul 28
variable "spot_priority" {
  type        = string
  description = "whether to do Spot instances, or Regular"
  default     = "Regular"
}

variable "spot_eviction_policy" {
  type        = string
  description = "what happens to spot instances that get evicted"
  default     = "Deallocate"
}

variable "vmss_os_image_id" {
  type        = string
  description = "Azure OS custom image id for VMSS instances"
  default     = ""
}

variable "vmss_os_image_publisher" {
  type        = string
  description = "Azure OS image publisher"
  default     = "Canonical"
}

variable "vmss_os_image_offer" {
  type        = string
  description = "Specifies the offer of the image used to create the virtual machines"
  default     = "UbuntuServer"
}

variable "vmss_os_image_sku" {
  type        = string
  description = "Specifies the SKU of the image used to create the virtual machines"
  default     = "16.04-LTS"
}

variable "vmss_os_image_version" {
  type        = string
  description = "Specifies the version of the image used to create the virtual machines"
  default     = "latest"
}

variable "nomad_az_vm_size" {
  type        = string
  description = "Specifies the size of Azure virtual machines in a scale set"
  default     = "Standard_A0"
}

variable "az_rg_name" {
  type        = string
  description = "Azure resource group name for specific environment"
}