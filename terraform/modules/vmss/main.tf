# VMSS with Loadbalancer
# TODO: Update to "azurerm_linux_virtual_machine_scale_set" resource  
resource "azurerm_virtual_machine_scale_set" "az_signals_vmss" {
  # Create only if 'az_vmss_uses_lb' is true
  count = var.az_vmss_uses_lb == "true" ? 1 : 0

  name                = "uphs_${var.env}_${var.app}_vmss"
  resource_group_name = var.az_rg_name
  location            = var.az_location
  upgrade_policy_mode = "Manual"
#  priority            = var.spot_priority
#  eviction_policy     = var.spot_eviction_policy

  sku {
    name     = var.az_instance_size
    tier     = var.az_instance_tier
    capacity = var.az_vmss_capacity
  }

  os_profile {
    computer_name_prefix = "${var.app}-${var.env}"
    admin_username       = var.admin_name
    custom_data          = var.custom_data
  }

  os_profile_linux_config {
    disable_password_authentication = true

    ssh_keys {
      path     = "/home/${var.admin_name}/.ssh/authorized_keys"
      key_data = file(var.admin_ssh_pub_key_file)
    }
  }

  network_profile {
    name                      = "uphs_${var.env}_${var.app}_nw_profile"
    primary                   = true
    network_security_group_id = var.az_nsg_id

    ip_configuration {
      name                                   = "uphs_${var.env}_${var.app}_ipconfig"
      primary                                = true
      subnet_id                              = var.az_subnet_id
      load_balancer_backend_address_pool_ids = var.az_lb_bak_addr_pool_ids

      #load_balancer_inbound_nat_rules_ids     = ["${var.az_lb_in_nat_rules_ids}"]
      application_security_group_ids = var.az_app_sec_grp_ids
    }
  }

  storage_profile_os_disk {
    name              = ""
    managed_disk_type = var.vmss_managed_disk_type
    create_option     = "FromImage"
    os_type           = "linux"
  }

  storage_profile_data_disk {
    lun               = 0
    create_option     = "Empty"
    disk_size_gb      = var.vmss_data_disk_size
    managed_disk_type = var.vmss_managed_disk_type
  }

  storage_profile_image_reference {
    id        = var.vmss_os_image_id
    publisher = var.vmss_os_image_publisher
    offer     = var.vmss_os_image_offer
    sku       = var.vmss_os_image_sku
    version   = var.vmss_os_image_version
  }

  tags = {
    application = var.app
    environment = var.env
  }
}

# VMSS without Loadbalancer
# TODO: Update to "azurerm_linux_virtual_machine_scale_set" resource  
resource "azurerm_virtual_machine_scale_set" "az_signals_vmss_no_lb" {
  # Create only if 'az_vmss_uses_lb' is true
  count = var.az_vmss_uses_lb == "false" ? 1 : 0

  name                = "uphs_${var.env}_${var.app}_vmss"
  resource_group_name = var.az_rg_name
  location            = var.az_location
  upgrade_policy_mode = "Manual"
  priority            = "Low"
  eviction_policy     = var.spot_eviction_policy

  sku {
    name     = var.az_instance_size
    tier     = var.az_instance_tier
    capacity = var.az_vmss_capacity
  }

  os_profile {
    computer_name_prefix = "${var.app}-${var.env}"
    admin_username       = var.admin_name
    custom_data          = var.custom_data
  }

  os_profile_linux_config {
    disable_password_authentication = true

    ssh_keys {
      path     = "/home/${var.admin_name}/.ssh/authorized_keys"
      key_data = file(var.admin_ssh_pub_key_file)
    }
  }

  network_profile {
    name                      = "uphs_${var.env}_${var.app}_nw_profile"
    primary                   = true
    network_security_group_id = var.az_nsg_id

    ip_configuration {
      name                           = "uphs_${var.env}_${var.app}_ipconfig"
      primary                        = true
      subnet_id                      = var.az_subnet_id
      application_security_group_ids = var.az_app_sec_grp_ids
    }
  }

  storage_profile_os_disk {
    name              = ""
    managed_disk_type = var.vmss_managed_disk_type
    create_option     = "FromImage"
    os_type           = "linux"
  }

  storage_profile_data_disk {
    lun               = 0
    create_option     = "Empty"
    disk_size_gb      = var.vmss_data_disk_size
    managed_disk_type = var.vmss_managed_disk_type
  }

  storage_profile_image_reference {
    id        = var.vmss_os_image_id
    publisher = var.vmss_os_image_publisher
    offer     = var.vmss_os_image_offer
    sku       = var.vmss_os_image_sku
    version   = var.vmss_os_image_version
  }

  tags = {
    application = var.app
    environment = var.env
  }
}


