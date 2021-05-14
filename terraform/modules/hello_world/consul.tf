module "consul_setup" {
  source = "../consul_setup"

  client_secret     = var.client_secret
  env               = var.env
  az_rg_name        = var.az_rg_name
  consul_agent_mode = "client"
  consul_enc_key    = var.consul_enc_key
}

