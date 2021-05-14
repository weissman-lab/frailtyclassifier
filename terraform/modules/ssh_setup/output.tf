output "ha_nodes_ssh" {
  value = file("${path.module}/templates/ha-ssh-setup.sh")
}

output "other_nodes_ssh" {
  value = file("${path.module}/templates/other-ssh-setup.sh")
}

