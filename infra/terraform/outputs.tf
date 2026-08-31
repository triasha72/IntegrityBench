output "service_url" {
  value = aws_apprunner_service.service.service_url
}

output "model_bucket" {
  value = aws_s3_bucket.models.id
}

output "review_table" {
  value = aws_dynamodb_table.reviews.name
}
