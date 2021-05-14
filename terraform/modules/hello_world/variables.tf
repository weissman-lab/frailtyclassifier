variable "server" {
  default     = true
  description = ""
}

variable "client" {
  default     = false
  description = ""
}

variable "quay_auth" {
  description = ""
}

variable "hello_world_version" {
  description = ""
}

variable "client_secret" {
  type        = string
  description = "Azure client secret"
}

variable "env" {
  type        = string
  description = "Environment name"
}

variable "az_rg_name" {
  type        = string
  description = "Azure resource group name for specific environment"
}

variable "az_location" {
  type        = string
  description = "Azure location"
}

variable "hello_world_subnet_id" {
  type        = string
  description = "Azure subnet ID of subnet for hello_world VMSS"
}

variable "hello_world_vmss_capacity" {
  type        = string
  description = "Provides the number of VMs for hello_world cluster"
}

variable "hello_world_az_vm_size" {
  type        = string
  description = "Specifies the size of Azure virtual machines in a scale set"
}

variable "hello_world_az_vm_tier" {
  type        = string
  description = "Specifies the tier of virtual machines in a scale set"
}

variable "hello_world_az_managed_disk_type" {
  type        = string
  description = "Specifies the type of managed disk to create"
}

variable "hello_world_az_disk_size_gb" {
  type        = string
  description = "Specifies the size of the Azure managed disk to create in gigabytes"
}

variable "hello_world_vmss_os_image_publisher" {
  type        = string
  description = "Specifies the image publisher"
}
variable "hello_world_vmss_os_image_offer" {
  type        = string
  description = "Specifies the image distribution"
}
variable "hello_world_vmss_os_image_sku" {
  type        = string
  description = "Specifies the sku"
}
variable "hello_world_vmss_os_image_version" {
  type        = string
  description = "Specifies the VM image version"
}

variable "consul_version" {
  type        = string
  description = "Version of Consul"
}

variable "consul_template_version" {
  type        = string
  description = "consul-template version to use"
}

variable "consul_enc_key" {
  type        = string
  description = "Base64 encoded random 16-byte string for Consul gossip encryption"
}

variable "allow_traffic_from_cidrs" {
  type        = list(string)
  description = "List of CIDRs to allow traffic from in the network security group rules"
  default     = []
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

variable "depends_on_modules" {
  default = []
  type    = list(string)
}

variable "enable_script_checks" {
  default     = false
  description = ""
}

variable "consul_root_ca" {
  type        = string
  description = "path to Consul-generated root certificate authority for Consul TLS"
}

variable "consul_root_ca_key" {
  type        = string
  description = "path to Consul-generated root certificate authority private key for Consul TLS"
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


