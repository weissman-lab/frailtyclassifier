# This 'env' module servers the following purposes:
# - Single source of truth for configuration details related to existing infrastructure
#   such as vnets, subnets, resource group, location etc.
# - Global environment specifc configuration and resource provisioning

locals {
  # Pre-existing resources already created in the Azure environment
  signals_rg_name             = "use2-uphs-pennsignals-prd-net-rg"
  signals_vnet_name           = "use2-uphs-pennsignals-prd-vnet"
  signals_rt_name             = "use2-uphs-pennsignals-prd-net-udr"
  signals_vnet_lb_subnet_name = "use2-uphs-pennsignals-lb-sub-1"
}

# Existing Azure resource group
data "azurerm_resource_group" "signals_rg" {
  name = local.signals_rg_name
}

# Existing Azure VNET information
data "azurerm_virtual_network" "signals_vnet" {
  name                = local.signals_vnet_name
  resource_group_name = local.signals_rg_name
}

# Existing Azure Loadbalancer subnet information
data "azurerm_subnet" "signals_lb_subnet" {
  name                 = local.signals_vnet_lb_subnet_name
  virtual_network_name = local.signals_vnet_name
  resource_group_name  = local.signals_rg_name
}

# Existing Azure route table
data "azurerm_route_table" "signals_rt" {
  name                = local.signals_rt_name
  resource_group_name = local.signals_rg_name
}

# Create the resource group for the environment being provisioned
resource "azurerm_resource_group" "signals_env_rg" {
  name     = "use2-uphs-pennsignals-ds-${var.env}-rg"
  location = data.azurerm_resource_group.signals_rg.location

  tags = {
    environment = var.env
  }
}

