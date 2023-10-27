library(tidyverse)


# ---------------------------------------------------------------------------- #
#                              Initial parameters                              #
# ---------------------------------------------------------------------------- #
N_TRAJ <- 7
NUMS <- c(1, 2, 3, 5, 10, 25, 50, 100, 1000) # 9 total
# NUMS <- c(NUMS[[3]], NUMS[[9]])

numbers <- map(
    NUMS,
    function(n) { str_pad(n, 4, pad="0")}
)

file_names <- map(
    numbers,
    function(m) { str_glue("out_", m, ".csv") }
)

command <- str_glue("./build/csv ", N_TRAJ)
system(command)

# Read the content of out_XXXX.csv and pivot the trajectories into a single column
mk_long_df <- function(pos) {
    df <- read_csv(file_names[[pos]])

    # # Now let's rotate this boy
    n_col = ncol(df)

    df_long <- df |>
        pivot_longer(
            names_to="trial_number",
            values_to="spot_price",
            cols=2:n_col
        )

    df_long
}

library(viridis)
library(scales)

colors <- c("black", "red")

for (i in seq(NUMS)) {
    df <- mk_long_df(i)
    g <- ggplot() +
        geom_point(data=df, mapping=aes(t, spot_price, col=trial_number)) +
        geom_line(data=df, mapping=aes(t, spot_price, col=trial_number)) +
        scale_color_viridis(discrete=TRUE, option="magma") +
        theme_minimal() +
        theme(legend.position = "none")

    ggsave(str_glue("traj_", NUMS[[i]], ".png"), g)
}

# g

# for (i in seq(NUMS)) {
#     print(i)
#     print(colors[i])
#     df <- mk_long_df(i)
#     g <- g +
#         geom_point(data=df, mapping=aes(t, spot_price, group_by=trial_number), color=colors[[i]]) +
#         geom_line(data=df, mapping=aes(t, spot_price, group_by=trial_number), color=colors[[i]])
# }

# g

df <- mk_long_df(1)
g <- g +
    geom_point(data=df, mapping=aes(t, spot_price)) +
    geom_line(data=df, mapping=aes(t, spot_price, group=trial_number)) +
    # scale_color_viridis(discrete=TRUE, option="magma") +
    theme_minimal() +
    theme(legend.position = "none")

df <- mk_long_df(2)
g <- g +
    geom_point(data=df, mapping=aes(t, spot_price), size=0.1, color="black") +
    geom_line(data=df, mapping=aes(t, spot_price, group=trial_number), color="red", alpha=0.5)

g

# ggsave("traj.png", g)