library(tidyverse)
library(lubridate)
library(grid)
library(dplyr)
library(AER)
library(stargazer)
library(ipumsr)
library(readr)

pums <- read.table(paste("QOB.raw",sep=""),
                   header           = FALSE,
                   stringsAsFactors = FALSE)
colnames(pums)[c(1,2,4,5,6,9:13,16,18,19:21,24,25,27)] <- c("AGE", "AGEQ", "EDUC", "ENOCENT","ESOCENT", "LWKLYWGE", "MARRIED", "MIDATL", "MT", "NEWENG", "CENSUS", "QOB", "RACE", "SMSA", "SOATL", "WNOCENT", "WSOCENT", "YOB")
pums <- as_tibble(pums)
pums

pums %>%
  mutate(cohort = factor(1*(YOB<=39 & YOB >=30) +
                           2*(YOB<=49 & YOB >=40),
                         levels=c(1,2), labels=c("30-39","40-49")) ) -> pums

# Add proper dates
pums$date <- ymd(paste("19", pums$YOB, pums$QOB * 3, sep=""), truncated = 2)

pums_aggregated <- pums %>%
  group_by(date, QOB) %>%
  summarise(
    EDUC = mean(EDUC),
    LWKLYWGE = mean(LWKLYWGE)
    )

# Time series graphics
ggplot(pums_aggregated, aes(x = date, y = EDUC, label = QOB)) +
  geom_line() +
  geom_label(data = subset(pums_aggregated, QOB == "1"), fill = "red", color = "white", label.r = unit(0.1, "in")) +
  geom_label(data = subset(pums_aggregated, QOB != "1"), fill = "black", color = "white", label.r = unit(0.1, "in")) +
  scale_color_manual(values = c("red", "black", "black", "black")) +
  ggtitle("Average education by quarter of birth") +
  labs(x = "Year of birth", y = "Years of education", color = "")

ggplot(pums_aggregated, aes(x = date, y = LWKLYWGE , label = QOB)) +
  geom_line() +
  geom_label(data = subset(pums, QOB == "1"), fill = "red", color = "white", label.r = unit(0.1, "in")) +
  geom_label(data = subset(pums, QOB != "1"), fill = "black", color = "white", label.r = unit(0.1, "in")) +
  scale_color_manual(values = c("red", "black", "black", "black")) +
  ggtitle("Average weekly wage by quarter of birth") +
  labs(x = "Year of birth", y = "Log weekly earnings", color = "")

pums %>%
  filter(cohort == "30-39") -> pums.tab5

## Table 5 analysis
exo2 =  "AGEQ +  I(AGEQ^2)"
exo3 = "RACE + MARRIED + SMSA + NEWENG + MIDATL + ENOCENT +
        WNOCENT + SOATL + ESOCENT + WSOCENT + MT"

reg_wage.etape1 <- ivreg(EDUC ~ QOB * YOB, data = pums.tab5)
pums.tab5$predicted <- predict(reg_wage.etape1)

reg_wage_a.MCO <- lm(LWKLYWGE ~ EDUC + YOB, data = pums.tab5)

reg_wage_a.TSLS <- lm(LWKLYWGE ~ predicted + YOB, data = pums.tab5)
names(reg_wage_a.TSLS$coefficients)[2] <- "EDUC"

reg_wage_b.MCO <- lm(LWKLYWGE ~ EDUC + YOB + AGE + I(AGE^2), data = pums.tab5)

reg_wage_b.TSLS <- lm(LWKLYWGE ~ predicted + YOB + AGE + I(AGE^2), data = pums.tab5)
names(reg_wage_b.TSLS$coefficients)[2] <- "EDUC"

reg_wage_c.MCO <- lm(LWKLYWGE ~ EDUC + YOB + RACE + SMSA + MARRIED + NEWENG + MIDATL + ENOCENT +  WNOCENT + SOATL + ESOCENT + WSOCENT + MT, data = pums.tab5)

reg_wage_c.TSLS <- lm(LWKLYWGE ~ predicted + YOB + RACE + SMSA + MARRIED + NEWENG + MIDATL + ENOCENT +  WNOCENT + SOATL + ESOCENT + WSOCENT + MT, data = pums.tab5)
names(reg_wage_c.TSLS$coefficients)[2] <- "EDUC"

reg_wage_d.MCO <- lm(LWKLYWGE ~ EDUC + YOB + RACE + SMSA + MARRIED + AGE + I(AGE^2) + NEWENG + MIDATL + ENOCENT +  WNOCENT + SOATL + ESOCENT + WSOCENT + MT, data = pums.tab5)

reg_wage_d.TSLS <- lm(LWKLYWGE ~ predicted + YOB + RACE + SMSA + MARRIED + AGE + I(AGE^2) + NEWENG + MIDATL + ENOCENT +  WNOCENT + SOATL + ESOCENT + WSOCENT + MT, data = pums.tab5)
names(reg_wage_d.TSLS$coefficients)[2] <- "EDUC"

table = stargazer(reg_wage_a.MCO, reg_wage_b.TSLS, reg_wage_b.MCO, reg_wage_b.TSLS, reg_wage_c.MCO, reg_wage_c.TSLS, reg_wage_d.MCO, reg_wage_d.TSLS,
                   dep.var.caption="",dep.var.labels="",
                   omit.table.layout = "n", star.cutoffs = NA,keep.stat=c("rsq","n"),
                   no.space=TRUE, digits=4,
                   header=FALSE,
                   keep=c("EDUC", "RACE", "SMSA", "MARRIED", "AGE", "I(AGE^2)"),
                   column.labels = c("OLS", "TSLS"),
                   title="OLS and TSLS Estimates of the Return to Education for Men Born 1930-1939: 1980 Census", type="text"
)

pums.tab5$wald_dum <- (pums.tab5$QOB == 3 | pums.tab5$QOB == 4) * 1
