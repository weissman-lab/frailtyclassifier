# Random 16-byte string for Consul gossip encryption
resource "random_id" "consul_enc_key" {
  byte_length = 16

  keepers = {
    env = var.env_name
  }
}

module "signals_env" {

  source = "./modules/env"

  env            = var.env_name
  env_ip_c_class = var.env_ip_c_class
}


module "hello_world" {

  source = "./modules/hello_world"

  client_secret      = var.client_secret
  env                = var.env_name
  #az_rg_name         = module.signals_env.env_rg_name
  az_rg_name         = var.az_rg_name
  az_location        = module.signals_env.az_location
  spot_priority      = var.spot_priority
  spot_eviction_policy      = var.spot_eviction_policy

  # hello_world_subnet_id = module.signals_env.hello_world_subnet_id
  # hello_world_subnet_id = module.signals_env.lb_subnet_id
  # subnet_id      = module.signals_env.lb_subnet_id

  hello_world_subnet_id = module.signals_env.lb_subnet_id

  allow_traffic_from_cidrs = [
    module.signals_env.lb_subnet_id,
  ]

  hello_world_vmss_capacity        = var.hello_world_capacity
  hello_world_az_vm_size           = var.hello_world_az_vm_size
  hello_world_az_vm_tier           = var.hello_world_az_vm_tier
  hello_world_az_managed_disk_type = var.hello_world_az_managed_disk_type
  hello_world_az_disk_size_gb      = var.hello_world_az_disk_size_gb
  admin_ssh_pub_key_file        = var.admin_ssh_pub_key_file

  hello_world_vmss_os_image_publisher = var.vmss_os_image_publisher
  hello_world_vmss_os_image_offer     = var.vmss_os_image_offer
  hello_world_vmss_os_image_sku       = var.vmss_os_image_sku
  hello_world_vmss_os_image_version   = var.vmss_os_image_version

  hello_world_version = var.hello_world_version
  server           = true
  client           = false
  quay_auth        = var.quay_auth

  consul_version          = var.consul_version
  consul_template_version = var.consul_template_version
  consul_enc_key          = random_id.consul_enc_key.b64_std
  consul_root_ca          = var.consul_root_ca
  consul_root_ca_key      = var.consul_root_ca_key
}