resource "aws_s3_bucket" "models" {
  bucket_prefix = "${var.name}-models-"
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_dynamodb_table" "reviews" {
  name         = "${var.name}-reviews"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "case_id"
  attribute {
    name = "case_id"
    type = "S"
  }
  point_in_time_recovery { enabled = true }
  server_side_encryption { enabled = true }
}

data "aws_iam_policy_document" "instance_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["tasks.apprunner.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "instance" {
  name_prefix        = "${var.name}-instance-"
  assume_role_policy = data.aws_iam_policy_document.instance_assume.json
}

data "aws_iam_policy_document" "runtime" {
  statement {
    actions   = ["s3:GetObject", "s3:GetObjectVersion"]
    resources = ["${aws_s3_bucket.models.arn}/*"]
  }
  statement {
    actions   = ["dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:Scan"]
    resources = [aws_dynamodb_table.reviews.arn]
  }
}

resource "aws_iam_role_policy" "runtime" {
  role   = aws_iam_role.instance.id
  policy = data.aws_iam_policy_document.runtime.json
}

data "aws_iam_policy_document" "ecr_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["build.apprunner.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecr" {
  name_prefix        = "${var.name}-ecr-"
  assume_role_policy = data.aws_iam_policy_document.ecr_assume.json
}

resource "aws_iam_role_policy_attachment" "ecr" {
  role       = aws_iam_role.ecr.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

resource "aws_apprunner_auto_scaling_configuration_version" "service" {
  auto_scaling_configuration_name = var.name
  max_concurrency                 = var.max_concurrency
  max_size                        = 5
  min_size                        = 1
}

resource "aws_apprunner_service" "service" {
  service_name = var.name
  source_configuration {
    auto_deployments_enabled = false
    authentication_configuration { access_role_arn = aws_iam_role.ecr.arn }
    image_repository {
      image_identifier      = var.image_identifier
      image_repository_type = "ECR"
      image_configuration {
        port = "8000"
        runtime_environment_variables = {
          INTEGRITYBENCH_REGISTRY     = "s3://${aws_s3_bucket.models.id}/${var.registry_key}"
          INTEGRITYBENCH_REVIEW_TABLE = aws_dynamodb_table.reviews.name
        }
      }
    }
  }
  instance_configuration {
    cpu               = "1 vCPU"
    memory            = "2 GB"
    instance_role_arn = aws_iam_role.instance.arn
  }
  auto_scaling_configuration_arn = aws_apprunner_auto_scaling_configuration_version.service.arn
  health_check_configuration {
    path                = "/health"
    protocol            = "HTTP"
    healthy_threshold   = 1
    unhealthy_threshold = 3
  }
}

resource "aws_cloudwatch_metric_alarm" "http_5xx" {
  alarm_name          = "${var.name}-5xx"
  namespace           = "AWS/AppRunner"
  metric_name         = "5xxStatusResponses"
  statistic           = "Sum"
  period              = 60
  evaluation_periods  = 2
  threshold           = 5
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  dimensions          = { ServiceName = aws_apprunner_service.service.service_name }
}
