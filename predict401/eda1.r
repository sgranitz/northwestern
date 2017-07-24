# Assignment 1 is an exploratory data analysis with the
# objective to determine plausible reasons why the original
# study was not successful in predicting age based on physical
# characteristics. This assignment is the precursor for the
# following assignment.

# Load libraries
require(tidyverse)
require(Hmisc)
require(plyr)
require(readr)
require(gridExtra)
require(knitr)

# Preliminaries: Load data
# This data file is derived from a study of abalones in Tasmania.

# (a) Reading the files into R
mydata <- read_csv("C:/Users/sgran/Desktop/DataAnalysis1/abalones.csv")

# (b) Check "mydata" using str().
# (1036 observations of 8 variables should be noted.)
str(mydata)

# Clean names and factors
mydata$SEX <- as.factor(mydata$SEX)
mydata$SEX <- revalue(mydata$SEX, c("F"="Female", "I"="Infant", "M"="Male"))
mydata$CLASS <- as.factor(mydata$CLASS)

# (c) Calculate two new variables: VOLUME and RATIO
mydata <- mydata %>%
  mutate(VOLUME = LENGTH * DIAM * HEIGHT,
         RATIO = SHUCK / VOLUME)
attach(mydata)

# Create subsets by gender
mydata_f <- mydata[mydata$SEX == "Female", ]
mydata_i <- mydata[mydata$SEX == "Infant", ]
mydata_m <- mydata[mydata$SEX == "Male", ]

# (1)(a) Use summary() to obtain and present descriptive statistics
# from mydata
summary(mydata)

ggplot(mydata, aes(x = RINGS, y = SEX, color = CLASS)) +
  geom_jitter(size = 2) + 
  theme_bw()

ggplot(mydata, aes(x = LENGTH, y = DIAM, color = SEX, size = HEIGHT)) +
  geom_point(shape = 1) + 
  theme_bw()
  
ggplot(mydata, aes(x = RATIO, y = VOLUME, color = SEX)) +
  geom_point() +
  geom_hline(yintercept = median(mydata_f$VOLUME), color = "indianred") +
  geom_hline(yintercept = median(mydata_i$VOLUME), color = "forestgreen") +
  geom_hline(yintercept = median(mydata_m$VOLUME), color = "steelblue") +
  facet_grid(.~SEX) + 
  theme_bw()

# (1)(b) Generate a table of counts using SEX and CLASS.
sex_class_tbl <- mydata %>%
  select(SEX, CLASS) %>%
  table() %>%
  addmargins()
print(sex_class_tbl)

# Also, present a barplot of these data.
sex_class_tbl[(1:3), (1:5)] %>%
  barplot(main = "Comparison of Age and Class Proportions",
          ylab = "Frequency", 
          ylim = c(0, 160),
          xlab = "Gender Distribution by Class",
          beside = TRUE,
          col = c('indianred', 'forestgreen', 'midnightblue'),
          legend.text = c('Female', 'Infant', 'Male'),
          args.legend = list(x = 'topright'))
          
mydata %>%
  ggplot(aes(CLASS, color=SEX)) +
  geom_freqpoly(aes(group=SEX), stat="count", size=2) +
  theme_bw()

# (1)(c) Select a simple random sample of 200 observations from
# "mydata" and identify this sample as "work".
set.seed(123)

work <- mydata %>%
  sample_n(size = 200, replace = FALSE)
plot(work[, 2:6], col = "steelblue")

# (2)(a) Use "mydata" to plot WHOLE versus VOLUME.
work2 <- mydata %>%
  select(WHOLE, VOLUME)
plot(x = work2$VOLUME, xlab = 'Volume',
     y = work2$WHOLE, ylab = 'Whole weight',
     main = 'Whole weight, as a function of Volume',
     col = 'goldenrod')
     
# (2)(b) Use "mydata" to plot SHUCK versus WHOLE
work3 <- mydata %>%
  select(SHUCK, WHOLE) %>%
  mutate(RATIO = SHUCK / WHOLE) %>%
  arrange(desc(RATIO))

ratio <- max(work3["RATIO"])

plot(x = work3$WHOLE, xlab = "Whole weight",
     y = work3$SHUCK, ylab = "Shuck weight",
     main = 'Shuck weight, as a function of Whole weight',
     col = 'forestgreen')
abline(a = 0, b = ratio, lty = 2)

# (3)(a) Use "mydata" to present a display showing histograms,
# boxplots and Q-Q plots of RATIO differentiated by sex.

par(mfrow = c(3, 3))

hist(mydata_f$RATIO, xlim = c(0, 0.35), ylim = c(0, 120),
     main = "Female Ratio", xlab ="", col = "indianred")

hist(mydata_i$RATIO, xlim = c(0, 0.35), ylim = c(0, 120),
     main = "Infant Ratio", xlab ="", col = "lightgreen")

hist(mydata_m$RATIO, xlim = c(0, 0.35), ylim = c(0, 120),
     main = "Male Ratio", xlab ="", col = "steelblue")
     
f_out <- boxplot(mydata_f$RATIO, ylim = c(0, 0.35), 
                 main = "Female Ratio", col = "indianred")$out

i_out <- boxplot(mydata_i$RATIO, ylim = c(0, 0.35),
                 main = "Infant Ratio", col = "lightgreen")$out

m_out <- boxplot(mydata_m$RATIO, ylim = c(0, 0.35),
                 main = "Male Ratio", col = "steelblue")$out

qqnorm(mydata_f$RATIO, ylim = c(0, 0.35), main = "Female Ratio",
       col = "indianred")
qqline(mydata_f$RATIO)

qqnorm(mydata_i$RATIO, ylim = c(0, 0.35), main = "Infant Ratio",
       col = "lightgreen")
qqline(mydata_i$RATIO)

qqnorm(mydata_m$RATIO, ylim = c(0, 0.35), main = "Male Ratio",
       col = "steelblue")
qqline(mydata_m$RATIO)

par(mfrow = c(1, 1))

ggplot(mydata, aes(x = RATIO, y = SEX, color = SEX)) +
  geom_jitter() + 
  theme_bw()

# (3)(b) Using the boxplots, identify and describe the abalones that are outliers.
out <- rbind(mydata_f %>% filter(RATIO %in% f_out),
             mydata_i %>% filter(RATIO %in% i_out),
             mydata_m %>% filter(RATIO %in% m_out))

out[c("SEX", "CLASS", "VOLUME", "RATIO")]

ggplot(out, aes(x = RATIO, y = CLASS, color = SEX)) +
  geom_jitter(size = 3) + 
  theme_bw()

# (4)(a) With "mydata," display two separate sets of side-by-side
# boxplots for VOLUME and WHOLE differentiated by CLASS

grid.arrange(
  ggplot(mydata, aes(x = factor(CLASS), y = VOLUME, group = CLASS)) +
    geom_boxplot(aes(color = CLASS)) + 
    labs(x = "Class", y = "Volume") +
    theme_bw(),
  ggplot(mydata, aes(x = factor(CLASS), y = WHOLE, group = CLASS)) +
    geom_boxplot(aes(color = CLASS)) + 
    labs(x = "Class", y = "Whole") +
    theme_bw(),
  ggplot(mydata, aes(x = RINGS, y = VOLUME, color = SEX)) +
    geom_point() + 
    labs(x = "Rings", y = "Volume") + 
    theme_bw(),
  ggplot(mydata, aes(x = RINGS, y = WHOLE, color = SEX)) +
    geom_point() + 
    labs(x = "Rings", y = "Whole") + 
    theme_bw(),
  nrow = 2, top = "Compare Volume and Whole Weight"
)

# (5)(a) Use aggregate() with "mydata" to compute the mean values of
# VOLUME, SHUCK and RATIO for each combination of SEX and CLASS.
myagg <- aggregate(mydata[c('VOLUME', 'SHUCK', 'RATIO')],
                   by = list(SEX, CLASS), FUN = 'mean')

matrix(myagg$VOLUME, nrow = 3, 
       dimnames = list(unique(myagg$Group.1),
                       unique(myagg$Group.2)))

matrix(myagg$SHUCK, nrow = 3,
       dimnames = list(unique(myagg$Group.1),
                              unique(myagg$Group.2)))

matrix(myagg$RATIO, nrow = 3,
       dimnames = list(unique(myagg$Group.1),
                       unique(myagg$Group.2)))

# (5)(b) Present three graphs
out <- aggregate(RATIO ~ SEX + CLASS, data = mydata, FUN = 'mean')
ggplot(data = out, aes(x = CLASS, y = RATIO, group = SEX, color = SEX)) +
  geom_line() + 
  theme_bw() +
  geom_point(size = 4) +
  ggtitle("Plot of Mean RATIO versus CLASS for Three Sexes")

out <- aggregate(VOLUME ~ SEX + CLASS, data = mydata, FUN = 'mean')
ggplot(data = out, aes(x = CLASS, y = VOLUME, group = SEX, color = SEX)) +
  geom_line() + 
  theme_bw() +
  geom_point(size = 4) +
  ggtitle("Plot of Mean VOLUME versus CLASS for Three Sexes")

out <- aggregate(SHUCK ~ SEX + CLASS, data = mydata, FUN = 'mean')
ggplot(data = out, aes(x = CLASS, y = SHUCK, group = SEX, color = SEX)) +
  geom_line() + 
  theme_bw() +
  geom_point(size = 4) +
  ggtitle("Plot of Mean SHUCK versus CLASS for Three Sexes")
