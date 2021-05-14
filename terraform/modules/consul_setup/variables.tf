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

variable "consul_agent_mode" {
  type        = string
  description = "Consul agent mode i.e. server/client"
}

variable "consul_vmss_capacity" {
  type        = string
  description = "Consul cluster size"
  default     = "3"
}

variable "consul_enc_key" {
  type        = string
  description = "Consul gossip encryption key"
}

variable "enable_script_checks" {
  type        = string
  description = "Consul agent enable_script_checks true/false"
  default     = "false"
}

variable "enable_consul_ui" {
  type        = string
  description = "Enable Consul UI"
  default     = "true"
}

