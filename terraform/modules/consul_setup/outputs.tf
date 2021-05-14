output "consul_config" {
  value = data.template_file.consul_config.rendered
}

output "consul_service_config" {
  value = file("${path.module}/templates/consul.service")
}

