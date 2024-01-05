












library(tidyverse)
library(lubridate)
library(grid)
library(dplyr)
library(AER)
library(stargazer)
library(ipumsr)
library(readr)

pums <- read.table(paste("file:///home/smouda/Documents/Doc_AMU/COURS/R/Projet_R/QOB.raw",sep=""),
                   header           = FALSE,
                   stringsAsFactors = FALSE)
colnames(pums)[c(1,2,4,5,6,9:13,16,18,19:21,24,25,27)] <- c("AGE", "AGEQ", "EDUC", "ENOCENT","ESOCENT", "LWKLYWGE", "MARRIED", "MIDATL", "MT", "NEWENG", "CENSUS", "QOB", "RACE", "SMSA", "SOATL", "WNOCENT", "WSOCENT", "YOB")
pums <- as_tibble(pums)

pums %>%
  mutate(cohort = factor(1*(YOB<=39 & YOB >=30) +
                           2*(YOB<=49 & YOB >=40),
                         levels=c(1,2), labels=c("30-39","40-49")) ) -> pums

pums

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
  geom_label(data = subset(pums_aggregated, QOB == "1"), fill = "red", color = "white", label.r = unit(0.1, "in")) +
  geom_label(data = subset(pums_aggregated, QOB != "1"), fill = "black", color = "white", label.r = unit(0.1, "in")) +
  scale_color_manual(values = c("red", "black", "black", "black")) +
  ggtitle("Average weekly wage by quarter of birth") +
  labs(x = "Year of birth", y = "Log weekly earnings", color = "")

pums %>%
  filter(cohort == "30-39") -> pums.tab5


reg_wage.etape1 <- ivreg(EDUC ~ QOB * YOB, data = pums.tab5)
pums.tab5$predicted <- predict(reg_wage.etape1)

reg_wage_a.MCO <- lm(LWKLYWGE ~ EDUC + YOB, data = pums.tab5)

reg_wage_a.TSLS <- lm(LWKLYWGE ~ predicted + YOB, data = pums.tab5)
names(reg_wage_a.TSLS$coefficients)[2] <- "EDUC"

reg_wage_b.MCO <- lm(LWKLYWGE ~ EDUC + YOB + AGE + I(AGE^2), data = pums.tab5)

reg_wage_b.TSLS <- lm(LWKLYWGE ~ predicted + YOB + AGE + I(AGE^2), data = pums.tab5)
names(reg_wage_b.TSLS$coefficients)[2] <- "EDUC"

tableau1 = stargazer(reg_wage_a.MCO, reg_wage_a.TSLS,reg_wage_b.MCO, reg_wage_b.TSLS,
                   dep.var.caption="",dep.var.labels="",
                   omit.table.layout = "n", star.cutoffs = NA,keep.stat=c("rsq","n"),
                   no.space=TRUE, digits=4,
                   header=FALSE,
                   keep=c("EDUC", "RACE", "SMSA", "MARRIED", "AGE", "I(AGE^2)"),
                   column.labels = c("OLS", "TSLS", "OLS", "TSLS"),
                   title="OLS and TSLS Estimates of the Return to Education for Men Born 1930-1939: 1980 Census", type="text"
)

reg_wage_c.MCO <- lm(LWKLYWGE ~ EDUC + YOB + RACE + SMSA + MARRIED + NEWENG + MIDATL + ENOCENT +  WNOCENT + SOATL + ESOCENT + WSOCENT + MT, data = pums.tab5)

reg_wage_c.TSLS <- lm(LWKLYWGE ~ predicted + YOB + RACE + SMSA + MARRIED + NEWENG + MIDATL + ENOCENT +  WNOCENT + SOATL + ESOCENT + WSOCENT + MT, data = pums.tab5)
names(reg_wage_c.TSLS$coefficients)[2] <- "EDUC"

reg_wage_d.MCO <- lm(LWKLYWGE ~ EDUC + YOB + RACE + SMSA + MARRIED + AGE + I(AGE^2) + NEWENG + MIDATL + ENOCENT +  WNOCENT + SOATL + ESOCENT + WSOCENT + MT, data = pums.tab5)

reg_wage_d.TSLS <- lm(LWKLYWGE ~ predicted + YOB + RACE + SMSA + MARRIED + AGE + I(AGE^2) + NEWENG + MIDATL + ENOCENT +  WNOCENT + SOATL + ESOCENT + WSOCENT + MT, data = pums.tab5)
names(reg_wage_d.TSLS$coefficients)[2] <- "EDUC"

tableau2 = stargazer(reg_wage_c.MCO, reg_wage_c.TSLS, reg_wage_d.MCO, reg_wage_d.TSLS,
                   dep.var.caption="",dep.var.labels="",
                   omit.table.layout = "n", star.cutoffs = NA,keep.stat=c("rsq","n"),
                   no.space=TRUE, digits=4,
                   header=FALSE,
                   keep=c("EDUC", "RACE", "SMSA", "MARRIED", "AGE", "I(AGE^2)"),
                   column.labels = c("OLS", "TSLS", "OLS", "TSLS"),
                   title="OLS and TSLS Estimates of the Return to Education for Men Born 1930-1939: 1980 Census", type="text"
)

pums.tab5$wald_dum <- (pums.tab5$QOB == 3 | pums.tab5$QOB == 4) * 1

ddi <- read_ipums_ddi(paste("/home/smouda/Téléchargements/usa_00001.xml",sep=""))
data_ipums <- read_ipums_micro(ddi, data_file =
                              paste("/home/smouda/Téléchargements/usa_00001.dat",sep=""))

data_ipums <- data_ipums %>%
  filter(INCWAGE != 0)


data_ipums %>%
  mutate(
    LWKLWGE = log((INCWAGE/52), base = exp(1)),
    cohort = factor(1*(BIRTHYR<=1939 & BIRTHYR >=1930) +
                           2*(BIRTHYR<=1949 & BIRTHYR >=1940),
                         levels=c(1,2), labels=c("1930-1939","1940-1949")) ) -> data_ipums


data_ipums

data_ipums %>%
  filter(cohort == "1930-1939") -> data_ipums.tab5

data_ipums.tab5

reg.etape1 <- ivreg(EDUC ~ BIRTHQTR * BIRTHYR, data = data_ipums.tab5)
data_ipums.tab5$predicted <- predict(reg.etape1)

reg_a.MCO <- lm(LWKLWGE ~ EDUC + BIRTHYR, data = data_ipums.tab5)

reg_a.TSLS <- lm(LWKLWGE ~ predicted + BIRTHYR, data = data_ipums.tab5)
names(reg_a.TSLS$coefficients)[2] <- "EDUC"

reg_b.MCO <- lm(LWKLWGE ~ EDUC + BIRTHYR + AGE + I(AGE^2), data = data_ipums.tab5)

reg_b.TSLS <- lm(LWKLWGE ~ predicted + BIRTHYR + AGE + I(AGE^2), data = data_ipums.tab5)
names(reg_b.TSLS$coefficients)[2] <- "EDUC"

tableau3 = stargazer(reg_a.MCO, reg_a.TSLS, reg_b.MCO, reg_b.TSLS,
                   dep.var.caption="",dep.var.labels="",
                   omit.table.layout = "n", star.cutoffs = NA,keep.stat=c("rsq","n"),
                   no.space=TRUE, digits=4,
                   header=FALSE,
                   keep=c("EDUC", "RACE", "AGE", "I(AGE^2)"),
                   column.labels = c("OLS", "TSLS", "OLS", "TSLS"),
                   title="OLS and TSLS Estimates of the Return to Education for Men Born 1930-1939: 1980 Ipums", type="text"
)


reg_c.MCO <- lm(LWKLWGE ~ EDUC + BIRTHYR + RACE, data = data_ipums.tab5)

reg_c.TSLS <- lm(LWKLWGE ~ predicted + BIRTHYR + RACE, data = data_ipums.tab5)
names(reg_c.TSLS$coefficients)[2] <- "EDUC"

reg_d.MCO <- lm(LWKLWGE ~ EDUC + BIRTHYR + RACE + AGE + I(AGE^2), data = data_ipums.tab5)

reg_d.TSLS <- lm(LWKLWGE ~ predicted + BIRTHYR + RACE + AGE + I(AGE^2), data = data_ipums.tab5)
names(reg_d.TSLS$coefficients)[2] <- "EDUC"

tableau4 = stargazer(reg_c.MCO, reg_c.TSLS, reg_d.MCO, reg_d.TSLS,
                      dep.var.caption="",dep.var.labels="",
                      omit.table.layout = "n", star.cutoffs = NA,keep.stat=c("rsq","n"),
                      no.space=TRUE, digits=4,
                      header=FALSE,
                      keep=c("EDUC", "RACE", "AGE", "I(AGE^2)"),
                      column.labels = c("OLS", "TSLS", "OLS", "TSLS"),
                      title="OLS and TSLS Estimates of the Return to Education for Men Born 1930-1939: 1980 Ipums", type="text"
)

