data "azurerm_client_config" "current" {
}

module "ssh_setup" {
  source = "../ssh_setup"
}

data "template_file" "cloudinit" {
  template = file("${path.module}/templates/cloud-init.conf")

  vars = {
    consul_version             = var.consul_version
    consul_config              = base64encode(module.consul_setup.consul_config)
    consul_service             = base64encode(module.consul_setup.consul_service_config)
    startup                    = base64encode(file("${path.module}/templates/startup.sh"))
    # check if we need this
    ha_ssh                     = base64encode(module.ssh_setup.ha_nodes_ssh)
    # nomad_version              = var.nomad_version
    nodes_ssh                  = base64encode(module.ssh_setup.other_nodes_ssh)
    consul_root_ca_content     = base64encode(file(var.consul_root_ca))
    consul_root_ca_key_content = base64encode(file(var.consul_root_ca_key))
  }
}

