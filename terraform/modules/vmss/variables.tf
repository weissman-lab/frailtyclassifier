variable "app" {
  type        = string
  description = "Application name"
}

variable "env" {
  type        = string
  description = "Environment name"
}

variable "az_rg_name" {
  type        = string
  description = "Azure resource group name"
}

variable "az_location" {
  type        = string
  description = "Azure location"
}

variable "az_instance_size" {
  type        = string
  description = "Azure VM instance size"
  default     = "Standard_A1"
}

variable "az_instance_tier" {
  type        = string
  description = "Azure VM instance tier"
  default     = "Standard"
}

variable "az_vmss_capacity" {
  type        = string
  description = "VMSS size"
}

variable "az_vmss_uses_lb" {
  type        = string
  description = "Create Loadbalancer for VMSS true/false"
  default     = "false"
}

variable "az_subnet_id" {
  type        = string
  description = "Azure subnet ID"
}

variable "az_lb_bak_addr_pool_ids" {
  type        = list(string)
  description = "List of Azure LB backend address pool id's"
  default     = []
}

/*variable "az_lb_in_nat_rules_ids" {
  type        = "list"
  description = "List of references to Azure LB inbound NAT rules"
  default     = []
}*/

variable "az_app_sec_grp_ids" {
  type        = list(string)
  description = "List of Azure application security group ids"
  default     = []
}

variable "az_nsg_id" {
  type        = string
  description = "ID of network security group to be associated"
  default     = ""
}

variable "vmss_data_disk_size" {
  type        = string
  description = "Azure data disk size in Gb"
  default     = "10"
}

variable "vmss_managed_disk_type" {
  type        = string
  description = "Azure managed disk type"
  default     = "Standard_LRS"
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

variable "admin_name" {
  type        = string
  description = "Azure VM instance admin account name"
  default     = "signals"
}

variable "admin_ssh_pub_key_file" {
  type        = string
  description = "Admin account SSH public key file path"
  default     = "foo.pub"
}

variable "custom_data" {
  type        = string
  description = "Custom data as cloud-init script"
  default     = ""
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
