output "rg_name" {
  description = "Existing Azure resource group name"
  value       = local.signals_rg_name
}

output "rg_id" {
  description = "Existing Azure resource group id"
  value       = data.azurerm_resource_group.signals_rg.id
}

output "az_location" {
  description = "Azure location to be used for all resources is based on location of existing resource group"
  value       = data.azurerm_resource_group.signals_rg.location
}

output "vnet_name" {
  description = "Existing Azure VNET name"
  value       = local.signals_vnet_name
}

output "vnet_id" {
  description = "Existing Azure VNET id"
  value       = data.azurerm_virtual_network.signals_vnet.id
}

output "rt_name" {
  description = "Existing Azure route table name"
  value       = local.signals_rt_name
}

output "rt_id" {
  description = "Existing Azure route table id"
  value       = data.azurerm_route_table.signals_rt.id
}

output "lb_subnet_name" {
  description = "Name of existing Azure subnet where the environment loadbalancer is to be placed"
  value       = local.signals_vnet_lb_subnet_name
}

output "lb_subnet_id" {
  description = "ID of existing Azure subnet where the environment loadbalancer is to be placed"
  value       = data.azurerm_subnet.signals_lb_subnet.id
}

output "env_rg_name" {
  description = "Azure resource group name for new Signals environment"
  value       = azurerm_resource_group.signals_env_rg.name
}

output "env_rg_id" {
  description = "Azure resource group id for new Signals environment"
  value       = azurerm_resource_group.signals_env_rg.id
}

// output "hello_world_subnet_cidr" {
//   description = "CIDR of hello_world subnet created"
//   value       = local.hello_world_subnet_cidr
// }

// output "hello_world_subnet_id" {
//   description = "ID of hello_world subnet created"
//   value       = azurerm_subnet.hello_world_subnet.id
// }