library(tidyverse)
library(lubridate)
library(grid)
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

# Time series graphics
ggplot(pums, aes(x = date, y = EDUC, label = QOB)) +
  geom_line() +
  geom_label(data = subset(pums, QOB == "1"), fill = "red", color = "white", label.r = unit(0.1, "in")) +
  geom_label(data = subset(pums, QOB != "1"), fill = "black", color = "white", label.r = unit(0.1, "in")) +
  scale_color_manual(values = c("red", "black", "black", "black")) +
  ggtitle("Average education by quarter of birth") +
  labs(x = "Year of birth", y = "Years of education", color = "")

ggplot(pums, aes(x = date, y = LWKLYWGE , label = QOB)) +
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

reg_wage.MCO <- lm(LWKLYWGE ~ RACE + MARRIED + SMSA + EDUC + AGEQ + I(AGEQ^2), data = pums.tab5)

reg_wage_a <- lm(LWKLYWGE ~ EDUC, data = pums.tab5)

reg_wage_b <- lm(LWKLYWGE ~ EDUC  + AGEQ + I(AGEQ^2), data = pums.tab5)

reg_wage_c <- lm(LWKLYWGE ~ EDUC + RACE + SMSA + MARRIED, data = pums.tab5)

reg_wage_d <- lm(LWKLYWGE ~ EDUC + RACE + SMSA + MARRIED + AGEQ + I(AGEQ^2), data = pums.tab5)

OLS_table <- stargazer(reg_wage_a,reg_wage_b,reg_wage_c,reg_wage_d,
                      dep.var.caption="",dep.var.labels="",
                      omit.table.layout = "n", star.cutoffs = NA,keep.stat=c("n","rsq"),
                      no.space=TRUE, digits=4,
                      header=FALSE,
                      column.labels=c("OLS", "OLS", "OLS", "OLS"), 
                      title="Table OLS", type="text"
                      )
