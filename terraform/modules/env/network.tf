locals {
  hello_world_subnet_cidr= "${var.env_ip_c_class}.224/28"
}

// # hello_world Subnet and route table association
// resource "azurerm_subnet" "hello_world_subnet" {
//   name                 = "uphs_${var.env}_hello_world_subnet"
//   resource_group_name  = local.signals_rg_name
//   virtual_network_name = local.signals_vnet_name
//   address_prefixes     = [local.hello_world_subnet_cidr]
// }

// resource "azurerm_subnet_route_table_association" "hello_world_subnet_rt" {
//   subnet_id      = azurerm_subnet.hello_world_subnet.id
//   route_table_id = data.azurerm_route_table.signals_rt.id
// }