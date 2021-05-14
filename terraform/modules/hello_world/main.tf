module "hello_world_vmss" {
  source = "../vmss"

  app                    = "hello"
  env                    = var.env
  az_rg_name             = var.az_rg_name
  az_location            = var.az_location
  az_subnet_id           = var.hello_world_subnet_id
  #az_subnet_id           = var.signals_lb_subnet
  az_nsg_id              = azurerm_network_security_group.hello_world_nsg.id
  az_app_sec_grp_ids     = [azurerm_application_security_group.hello_world_asg.id]
  az_vmss_capacity       = var.hello_world_vmss_capacity
  az_vmss_uses_lb        = "false"
  az_instance_size       = var.hello_world_az_vm_size
  az_instance_tier       = var.hello_world_az_vm_tier
  vmss_data_disk_size    = var.hello_world_az_disk_size_gb
  vmss_managed_disk_type = var.hello_world_az_managed_disk_type
  admin_name             = var.admin_name
  admin_ssh_pub_key_file = var.admin_ssh_pub_key_file
  custom_data            = data.template_file.cloudinit.rendered
  spot_priority          = var.spot_priority
  spot_eviction_policy   = var.spot_eviction_policy

  // vmss_os_image_id        = var.hello_world_vmss_os_image_id
  vmss_os_image_publisher = var.hello_world_vmss_os_image_publisher
  vmss_os_image_offer     = var.hello_world_vmss_os_image_offer
  vmss_os_image_sku       = var.hello_world_vmss_os_image_sku
  vmss_os_image_version   = var.hello_world_vmss_os_image_version
  
}