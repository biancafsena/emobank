pkgs <- c('tidyverse','yardstick','rmarkdown')
to_install <- pkgs[!pkgs %in% installed.packages()[,'Package']]
if(length(to_install) > 0) install.packages(to_install, repos='https://cloud.r-project.org')
