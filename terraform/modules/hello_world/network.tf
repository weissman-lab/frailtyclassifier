# Vault Application Security Group
resource "azurerm_application_security_group" "hello_world_asg" {
  name                = "uphs_${var.env}_hello_world_asg"
  location            = var.az_location
  resource_group_name = var.az_rg_name

  tags = {
    application = "hello_world"
    environment = var.env
  }
}

# Vault Network Security Group and Rules
resource "azurerm_network_security_group" "hello_world_nsg" {
  name                = "uphs_${var.env}_hello_world_nsg"
  location            = var.az_location
  resource_group_name = var.az_rg_name

  tags = {
    application = "hello_world"
    environment = var.env
  }
}

# Allow SSH traffic
resource "azurerm_network_security_rule" "hello_world_ssh_nsg_rule" {
  name                                       = "uphs_${var.env}_hello_world_ssh_nsg_rule"
  resource_group_name                        = var.az_rg_name
  network_security_group_name                = azurerm_network_security_group.hello_world_nsg.name
  priority                                   = 700
  direction                                  = "Inbound"
  access                                     = "Allow"
  protocol                                   = "Tcp"
  source_port_range                          = "*"
  source_address_prefixes                    = var.allow_traffic_from_cidrs
  destination_port_range                     = 22
  destination_application_security_group_ids = [azurerm_application_security_group.hello_world_asg.id]
}

# Allow Inbound HTTP traffic on port 5432
resource "azurerm_network_security_rule" "hello_world_http_api_nsg_rule" {
  name                                       = "uphs_${var.env}_hello_world_http_api_nsg_rule"
  resource_group_name                        = var.az_rg_name
  network_security_group_name                = azurerm_network_security_group.hello_world_nsg.name
  priority                                   = 900
  direction                                  = "Inbound"
  access                                     = "Allow"
  protocol                                   = "Tcp"
  source_port_range                          = "*"
  source_address_prefixes                    = var.allow_traffic_from_cidrs
  destination_port_range                     = 5432
  destination_application_security_group_ids = [azurerm_application_security_group.hello_world_asg.id]
}

