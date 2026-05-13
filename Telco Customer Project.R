install.packages("factoextra")
install.packages("caret")
install.packages("randomForest")
install.packages("caTools")

library(tidyverse)
library(cluster)
library(randomForest)
library(caTools)

# setwd("path/to/your/working/directory")

# load data
# Dataset: IBM Telco Customer Churn, available on Kaggle
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
telco_data <- read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# check dimensions
nrow(telco_data)
ncol(telco_data)

# check for missing values
colSums(is.na(telco_data))

# view churn distribution
telco_data %>%
  group_by(Churn) %>%
  summarize(count = n(),
            percentage = round(n() / nrow(telco_data) * 100, 2))


# data preparation for clustering
# create binary service indicators
telco_prepared <- telco_data %>%
  mutate(
    has_internet = ifelse(InternetService != "No", 1, 0),
    has_online_security = ifelse(OnlineSecurity == "Yes", 1, 0),
    has_tech_support = ifelse(TechSupport == "Yes", 1, 0),
    has_phone_service = ifelse(PhoneService == "Yes", 1, 0),
    contract_month_to_month = ifelse(Contract == "Month-to-month", 1, 0),
    contract_one_year = ifelse(Contract == "One year", 1, 0),
    contract_two_year = ifelse(Contract == "Two year", 1, 0)
  )

# subset data for clustering
clustering_data <- telco_prepared %>%
  select(tenure, has_internet, has_online_security, 
         has_tech_support, has_phone_service, contract_month_to_month, 
         contract_one_year, contract_two_year)

# standardize the data
clustering_scaled <- scale(clustering_data)

# verify scaling worked
round(colMeans(clustering_scaled), 5)
round(apply(clustering_scaled, 2, sd), 5)


# determine optimal number of clusters using elbow method
set.seed(2025)

elbow_results <- tibble(k = 1:10, wss = NA_real_)

for (i in 1:10) {
  kmeans_temp <- kmeans(clustering_scaled, centers = i, nstart = 25)
  elbow_results$wss[i] <- kmeans_temp$tot.withinss
}

elbow_results

# plot elbow curve
ggplot(data = elbow_results, aes(x = k, y = wss)) +
  geom_point(size = 3) +
  geom_line() +
  labs(title = "Elbow Method for Optimal k",
       x = "Number of Clusters (k)",
       y = "Within-Cluster Sum of Squares") +
  scale_x_continuous(breaks = 1:10) +
  theme_bw()


# run k-means with k = 4
K <- 4
set.seed(2025)

kmeans_model <- kmeans(clustering_scaled, centers = K, nstart = 25)

# check cluster sizes
kmeans_model$size

# add cluster assignments to data
telco_clustered <- telco_prepared %>%
  mutate(cluster = as.factor(kmeans_model$cluster))

# view cluster distribution
telco_clustered %>%
  group_by(cluster) %>%
  summarize(count = n(),
            percentage = round(n() / nrow(telco_clustered) * 100, 2))


# analyze cluster characteristics - tenure and churn
cluster_profile <- telco_clustered %>%
  group_by(cluster) %>%
  summarize(
    avg_tenure = round(mean(tenure), 2),
    avg_monthly_charges = round(mean(MonthlyCharges), 2),
    churn_rate = round(sum(Churn == "Yes") / n() * 100, 2),
    count = n()
  )

cluster_profile

# service adoption by cluster
service_profile <- telco_clustered %>%
  group_by(cluster) %>%
  summarize(
    pct_has_internet = round(sum(has_internet == 1) / n() * 100, 2),
    pct_online_security = round(sum(has_online_security == 1) / n() * 100, 2),
    pct_tech_support = round(sum(has_tech_support == 1) / n() * 100, 2),
    pct_phone_service = round(sum(has_phone_service == 1) / n() * 100, 2)
  )

service_profile

# contract type by cluster
contract_profile <- telco_clustered %>%
  group_by(cluster) %>%
  summarize(
    pct_month_to_month = round(sum(Contract == "Month-to-month") / n() * 100, 2),
    pct_one_year = round(sum(Contract == "One year") / n() * 100, 2),
    pct_two_year = round(sum(Contract == "Two year") / n() * 100, 2)
  )

contract_profile


# calculate customer lifetime value (CLV)
# CLV = Monthly Charges × Tenure in Months
telco_clv <- telco_clustered %>%
  mutate(
    total_charges_numeric = as.numeric(TotalCharges),
    CLV = MonthlyCharges * tenure,
    CLV_actual = ifelse(is.na(total_charges_numeric), 
                        MonthlyCharges * tenure, 
                        total_charges_numeric)
  )

# check for missing CLV values
sum(is.na(telco_clv$CLV))
sum(is.na(telco_clv$CLV_actual))

# CLV statistics by cluster
clv_profile <- telco_clv %>%
  group_by(cluster) %>%
  summarize(
    avg_clv = round(mean(CLV_actual), 2),
    median_clv = round(median(CLV_actual), 2),
    total_clv = round(sum(CLV_actual), 2),
    customer_count = n()
  )

clv_profile

# analyze churn impact by cluster
churn_clv_impact <- telco_clv %>%
  filter(Churn == "Yes") %>%
  group_by(cluster) %>%
  summarize(
    churned_customers = n(),
    total_clv_lost = round(sum(CLV_actual), 2),
    avg_clv_lost = round(mean(CLV_actual), 2)
  )

churn_clv_impact


# prepare data for predictive modeling
model_data <- telco_clv %>%
  mutate(
    churn_binary = ifelse(Churn == "Yes", 1, 0),
    gender_male = ifelse(gender == "Male", 1, 0),
    senior_citizen = SeniorCitizen,
    partner_yes = ifelse(Partner == "Yes", 1, 0),
    dependents_yes = ifelse(Dependents == "Yes", 1, 0),
    paperless_yes = ifelse(PaperlessBilling == "Yes", 1, 0)
  ) %>%
  select(churn_binary, cluster, tenure, MonthlyCharges, CLV_actual,
         gender_male, senior_citizen, partner_yes, dependents_yes,
         has_internet, has_online_security, has_tech_support, has_phone_service,
         contract_month_to_month, contract_one_year, contract_two_year,
         paperless_yes)

# remove any missing values
model_data <- model_data %>% drop_na()

# split data into train and test (80/20)
set.seed(2025)
data_split <- sample.split(model_data$churn_binary, SplitRatio = 0.8)

train_data <- model_data %>% filter(data_split == TRUE)
test_data <- model_data %>% filter(data_split == FALSE)

# verify split
nrow(train_data) / nrow(model_data)
nrow(test_data) / nrow(model_data)


# fit logistic regression model
logistic_model <- glm(churn_binary ~ ., 
                      data = train_data, 
                      family = "binomial")

summary(logistic_model)

# make predictions on test data
logistic_predictions <- predict(logistic_model, 
                                newdata = test_data, 
                                type = "response")

logistic_pred_binary <- ifelse(logistic_predictions > 0.5, 1, 0)

# evaluate logistic regression
logistic_accuracy <- mean(logistic_pred_binary == test_data$churn_binary)
logistic_accuracy

logistic_confusion <- table(predicted = logistic_pred_binary, 
                            actual = test_data$churn_binary)
logistic_confusion


# fit random forest model
set.seed(2025)

rf_model <- randomForest(as.factor(churn_binary) ~ ., 
                         data = train_data, 
                         ntree = 100,
                         importance = TRUE)

print(rf_model)

# feature importance
feature_imp <- importance(rf_model) %>%
  as.data.frame() %>%
  rownames_to_column("feature") %>%
  arrange(desc(MeanDecreaseGini))

feature_imp

# plot feature importance
ggplot(data = feature_imp, aes(x = reorder(feature, MeanDecreaseGini), 
                               y = MeanDecreaseGini)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance in Random Forest",
       x = "Feature",
       y = "Mean Decrease in Gini") +
  theme_bw()

# make predictions on test data
rf_predictions <- predict(rf_model, newdata = test_data, type = "response")
rf_pred_binary <- as.numeric(as.character(rf_predictions))

# evaluate random forest
rf_accuracy <- mean(rf_pred_binary == test_data$churn_binary)
rf_accuracy

rf_confusion <- table(predicted = rf_pred_binary, 
                      actual = test_data$churn_binary)
rf_confusion


# get churn probabilities from both models on full dataset
churn_probs_logistic <- predict(logistic_model, 
                                newdata = model_data, 
                                type = "response")

churn_probs_rf <- predict(rf_model, 
                          newdata = model_data, 
                          type = "prob")[, 2]

# calculate risk scores
risk_scores <- model_data %>%
  mutate(
    churn_prob_logistic = churn_probs_logistic,
    churn_prob_rf = churn_probs_rf,
    churn_prob_avg = (churn_probs_logistic + churn_probs_rf) / 2,
    clv_normalized = (CLV_actual - min(CLV_actual)) / 
      (max(CLV_actual) - min(CLV_actual)),
    risk_score = churn_prob_avg * clv_normalized
  )

# identify high-risk customers (top 25%)
high_risk_threshold <- quantile(risk_scores$risk_score, 0.75)

high_risk_customers <- risk_scores %>%
  filter(risk_score >= high_risk_threshold) %>%
  arrange(desc(risk_score))

nrow(high_risk_customers)
head(high_risk_customers, 20)

# high-risk customers by cluster
high_risk_analysis <- risk_scores %>%
  mutate(is_high_risk = risk_score >= high_risk_threshold) %>%
  group_by(cluster) %>%
  summarize(
    total_customers = n(),
    high_risk_count = sum(is_high_risk),
    high_risk_pct = round(sum(is_high_risk) / n() * 100, 2),
    total_clv_at_risk = round(sum(CLV_actual[is_high_risk], na.rm = TRUE), 2)
  )

high_risk_analysis

# export results
write_csv(cluster_profile, "cluster_profiles.csv")
write_csv(clv_profile, "clv_by_cluster.csv")
write_csv(churn_clv_impact, "churn_clv_impact.csv")
write_csv(high_risk_analysis, "high_risk_by_cluster.csv")
write_csv(risk_scores, "all_customers_risk_scores.csv")

# visualizations

# 1. Cluster size distribution
ggplot(data = cluster_profile, aes(x = cluster, y = count, fill = cluster)) +
  geom_col() +
  geom_text(aes(label = count), vjust = -0.5, size = 4) +
  labs(title = "Customer Distribution Across Clusters",
       x = "Cluster",
       y = "Number of Customers") +
  theme_bw() +
  theme(legend.position = "none")

# 2. Churn rate by cluster
ggplot(data = cluster_profile, aes(x = cluster, y = churn_rate, fill = cluster)) +
  geom_col() +
  geom_text(aes(label = paste0(churn_rate, "%")), vjust = -0.5, size = 4) +
  labs(title = "Churn Rate by Cluster",
       x = "Cluster",
       y = "Churn Rate (%)") +
  theme_bw() +
  theme(legend.position = "none")

# 3. Average tenure by cluster
ggplot(data = cluster_profile, aes(x = cluster, y = avg_tenure, fill = cluster)) +
  geom_col() +
  geom_text(aes(label = avg_tenure), vjust = -0.5, size = 4) +
  labs(title = "Average Tenure by Cluster",
       x = "Cluster",
       y = "Average Tenure (Months)") +
  theme_bw() +
  theme(legend.position = "none")

# 4. Average monthly charges by cluster
ggplot(data = cluster_profile, aes(x = cluster, y = avg_monthly_charges, fill = cluster)) +
  geom_col() +
  geom_text(aes(label = paste0("$", avg_monthly_charges)), vjust = -0.5, size = 4) +
  labs(title = "Average Monthly Charges by Cluster",
       x = "Cluster",
       y = "Average Monthly Charges ($)") +
  theme_bw() +
  theme(legend.position = "none")

# 5. CLV by cluster
ggplot(data = clv_profile, aes(x = cluster, y = total_clv, fill = cluster)) +
  geom_col() +
  geom_text(aes(label = paste0("$", round(total_clv/1000000, 2), "M")), vjust = -0.5, size = 4) +
  labs(title = "Total Customer Lifetime Value by Cluster",
       x = "Cluster",
       y = "Total CLV ($)") +
  theme_bw() +
  theme(legend.position = "none")

# 6. Revenue lost to churn by cluster
ggplot(data = churn_clv_impact, aes(x = cluster, y = total_clv_lost, fill = cluster)) +
  geom_col() +
  geom_text(aes(label = paste0("$", round(total_clv_lost/1000, 1), "K")), vjust = -0.5, size = 4) +
  labs(title = "Revenue Lost to Churn by Cluster",
       x = "Cluster",
       y = "Total CLV Lost ($)") +
  theme_bw() +
  theme(legend.position = "none")

# 7. Service adoption heatmap - reshape service profile for heatmap
service_data <- service_profile %>%
  pivot_longer(cols = -cluster, names_to = "service", values_to = "percentage") %>%
  mutate(service = gsub("pct_", "", service))

ggplot(data = service_data, aes(x = service, y = cluster, fill = percentage)) +
  geom_tile() +
  geom_text(aes(label = paste0(percentage, "%")), color = "white", size = 4, fontface = "bold") +
  scale_fill_gradient(low = "darkred", high = "darkgreen") +
  labs(title = "Service Adoption by Cluster (%)",
       x = "Service Type",
       y = "Cluster",
       fill = "Adoption %") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 8. Contract type distribution heatmap
contract_data <- contract_profile %>%
  pivot_longer(cols = -cluster, names_to = "contract_type", values_to = "percentage") %>%
  mutate(contract_type = gsub("pct_", "", contract_type))

ggplot(data = contract_data, aes(x = contract_type, y = cluster, fill = percentage)) +
  geom_tile() +
  geom_text(aes(label = paste0(percentage, "%")), color = "white", size = 4, fontface = "bold") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Contract Type Distribution by Cluster (%)",
       x = "Contract Type",
       y = "Cluster",
       fill = "Percentage") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 9. Churn vs retention by cluster
churn_summary <- telco_clv %>%
  group_by(cluster, Churn) %>%
  summarize(count = n())

ggplot(data = churn_summary, aes(x = cluster, y = count, fill = Churn)) +
  geom_col(position = "dodge") +
  labs(title = "Churn vs Retention by Cluster",
       x = "Cluster",
       y = "Number of Customers",
       fill = "Status") +
  scale_fill_manual(values = c("No" = "green", "Yes" = "red")) +
  theme_bw()

# 10. Average CLV per customer by cluster
clv_per_customer <- clv_profile %>%
  mutate(clv_per_cust = total_clv / customer_count)

ggplot(data = clv_per_customer, aes(x = cluster, y = clv_per_cust, fill = cluster)) +
  geom_col() +
  geom_text(aes(label = paste0("$", round(clv_per_cust, 0))), vjust = -0.5, size = 4) +
  labs(title = "Average CLV per Customer by Cluster",
       x = "Cluster",
       y = "Average CLV per Customer ($)") +
  theme_bw() +
  theme(legend.position = "none")

# 11. High-risk customers count by cluster
ggplot(data = high_risk_analysis, aes(x = cluster, y = high_risk_count, fill = cluster)) +
  geom_col() +
  geom_text(aes(label = high_risk_count), vjust = -0.5, size = 4) +
  labs(title = "High-Risk Customers by Cluster (Top 25%)",
       x = "Cluster",
       y = "Number of High-Risk Customers") +
  theme_bw() +
  theme(legend.position = "none")

# 12. High-risk percentage by cluster
ggplot(data = high_risk_analysis, aes(x = cluster, y = high_risk_pct, fill = cluster)) +
  geom_col() +
  geom_text(aes(label = paste0(high_risk_pct, "%")), vjust = -0.5, size = 4) +
  labs(title = "Percentage of High-Risk Customers by Cluster",
       x = "Cluster",
       y = "High-Risk Percentage (%)") +
  theme_bw() +
  theme(legend.position = "none")

# 13. Risk score distribution by cluster
ggplot(data = risk_scores, aes(x = risk_score, fill = cluster)) +
  geom_histogram(bins = 40, alpha = 0.7) +
  facet_wrap(~cluster) +
  labs(title = "Risk Score Distribution by Cluster",
       x = "Risk Score",
       y = "Frequency",
       fill = "Cluster") +
  theme_bw()

# 14. Scatter plot: Tenure vs Monthly Charges colored by cluster
ggplot(data = risk_scores, aes(x = tenure, y = MonthlyCharges, color = cluster, size = CLV_actual)) +
  geom_point(alpha = 0.5) +
  labs(title = "Tenure vs Monthly Charges by Cluster",
       x = "Tenure (Months)",
       y = "Monthly Charges ($)",
       color = "Cluster",
       size = "CLV") +
  theme_bw()

# 15. Model comparison - accuracy and confusion matrices visual
model_comparison <- tibble(
  Model = c("Logistic Regression", "Random Forest"),
  Accuracy = c(logistic_accuracy, rf_accuracy)
)

ggplot(data = model_comparison, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_col() +
  geom_text(aes(label = paste0(round(Accuracy * 100, 2), "%")), vjust = -0.5, size = 5) +
  ylim(0, 1) +
  labs(title = "Model Accuracy Comparison",
       x = "Model",
       y = "Accuracy") +
  theme_bw() +
  theme(legend.position = "none")

# 16. Logistic regression confusion matrix visualization
logistic_conf_data <- as.data.frame(logistic_confusion) %>%
  rename(predicted_class = predicted, actual_class = actual, count = Freq)

ggplot(data = logistic_conf_data, aes(x = predicted_class, y = actual_class, fill = count)) +
  geom_tile() +
  geom_text(aes(label = count), color = "white", size = 5, fontface = "bold") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Logistic Regression - Confusion Matrix",
       x = "Predicted",
       y = "Actual",
       fill = "Count") +
  theme_bw()

# 17. Random forest confusion matrix visualization
rf_conf_data <- as.data.frame(rf_confusion) %>%
  rename(predicted_class = predicted, actual_class = actual, count = Freq)

ggplot(data = rf_conf_data, aes(x = predicted_class, y = actual_class, fill = count)) +
  geom_tile() +
  geom_text(aes(label = count), color = "white", size = 5, fontface = "bold") +
  scale_fill_gradient(low = "lightgreen", high = "darkgreen") +
  labs(title = "Random Forest - Confusion Matrix",
       x = "Predicted",
       y = "Actual",
       fill = "Count") +
  theme_bw()

# 18. Churn probability distribution
ggplot(data = risk_scores, aes(x = churn_prob_avg, fill = as.factor(churn_binary))) +
  geom_histogram(bins = 40, alpha = 0.7) +
  labs(title = "Distribution of Predicted Churn Probability",
       x = "Churn Probability",
       y = "Frequency",
       fill = "Actual Churn") +
  scale_fill_manual(values = c("0" = "green", "1" = "red"), labels = c("0" = "No Churn", "1" = "Churned")) +
  theme_bw()

# 19. CLV distribution by cluster
ggplot(data = risk_scores, aes(x = cluster, y = CLV_actual, fill = cluster)) +
  geom_boxplot() +
  labs(title = "CLV Distribution by Cluster",
       x = "Cluster",
       y = "Customer Lifetime Value ($)") +
  theme_bw() +
  theme(legend.position = "none")

# 20. Churn rate vs CLV by cluster
cluster_summary <- cluster_profile %>%
  left_join(clv_profile, by = "cluster") %>%
  mutate(avg_clv_profile = total_clv / count)

ggplot(data = cluster_summary, aes(x = churn_rate, y = avg_clv_profile, size = count, color = cluster)) +
  geom_point(alpha = 0.6) +
  geom_text(aes(label = cluster), vjust = -1, size = 5) +
  labs(title = "Churn Rate vs Average CLV by Cluster",
       x = "Churn Rate (%)",
       y = "Average CLV per Customer ($)",
       color = "Cluster",
       size = "Customer Count") +
  theme_bw()