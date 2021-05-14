variable "env" {
  type        = string
  description = "Signals environment name"
}

variable "env_ip_c_class" {
  type        = string
  description = "The first three octets for the environment. Used to calulate subnets for different layers"
}

