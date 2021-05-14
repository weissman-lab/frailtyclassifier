data "azurerm_client_config" "current" {
}

data "template_file" "consul_config" {
  template = file(
    "${path.module}/templates/${var.consul_agent_mode}_config.json.tmpl",
  )

  vars = {
    bootstrap_expect     = var.consul_vmss_capacity
    enable_script_checks = var.enable_script_checks
    consul_enc_key       = var.consul_enc_key
    server               = var.consul_agent_mode == "server" ? "true" : "false"
    ui                   = var.enable_consul_ui
    az_subscription_id   = data.azurerm_client_config.current.subscription_id
    az_tenant_id         = data.azurerm_client_config.current.tenant_id
    az_client_id         = data.azurerm_client_config.current.client_id
    az_secret_access_key = var.client_secret
    az_rg_name           = var.az_rg_name
    az_consul_vmss_name  = "uphs_${var.env}_consul_vmss"
  }
}

