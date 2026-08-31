variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "name" {
  type    = string
  default = "integritybench"
}

variable "image_identifier" {
  description = "Immutable ECR image URI including digest or unique tag."
  type        = string
}

variable "registry_key" {
  type    = string
  default = "registry/registry.json"
}

variable "max_concurrency" {
  type    = number
  default = 50
}
