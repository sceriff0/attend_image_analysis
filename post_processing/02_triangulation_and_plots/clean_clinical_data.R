clinical_data <- read_delim("data/clinical_data.csv", delim=";")


id_list <- c(
    "22-S-000016",
    "21-S-000238", 
    "21-S-000113",
    "22-S-000005",
    "20-S-000234",
    "21-S-000188",
    "23-S-00022",
    "23-S-00064", 
    "21-S-000105",
    "19-S-000059"
  )



selected_clinical_data <- clinical_data %>% 
  rename(
    patient_id = `Codice caso studio IEO`
  ) %>% 
  filter(patient_id %in% id_list)



id_list_new <- c(
  "1818600",
  "35138",
  "1826315",
  "21-37601",
  "19E110099A4",
  "21S188",
  "23S22",
  "23S64", 
  "1923575",
  "B19-10215"
)

patient_id_map <- tibble(
  patient_id = id_list, 
  patient_id_new = id_list_new
)



clinical_data_clean <- clinical_data %>% 
  rename_with(~ gsub(" ", "_", tolower(.))) %>%
  rename_with(~ gsub("\\.", "", .)) %>% 
  rename(
    patient_id = codice_caso_studio_ieo,
    pfs_event = `pfs_event_(pd/death/no_event)`
  ) %>% 
  mutate(
    pfs_time = as.numeric(str_replace(pfs_time, ",", ".")),
    race = as.factor(race),
    status_of_disease = as.factor(status_of_disease),
    figo_stage = as.factor(figo_stage),
    pfs_event = as.factor(pfs_event)
  )



clinical_data_selected <- clinical_data_clean %>% 
  select(
    patient_id, treatment_arm, race, status_of_disease, figo_stage, pfs_event, pfs_time
  ) %>% 
  left_join(patient_id_map) %>% 
  filter(!is.na(patient_id_new)) %>% 
  select(-patient_id) %>% 
  rename(
    patient_id = patient_id_new
  )

status_dict <- c(
  "23S64" = "high",
  "35138" = "low",
  "19E110099A4" = "low",
  "1826315" = "low",
  "B19-10215" = "high",
  "21S188" = "high",
  "21-37601" = "low",
  "1923575" = "high",
  "23S22" = "high",
  "1818600" = "low"
)

# Assign based on dictionary
clinical_data_selected$aneuploidy <- status_dict[as.factor(clinical_data_selected$patient_id)]
clinical_data_selected <- clinical_data_selected %>% 
  mutate(aneuploidy = as.factor(aneuploidy))

rm(status_dict, clinical_data_clean, patient_id_map, id_list_new, clinical_data, id_list, selected_clinical_data)

#clinical_data_joined <- clinical_data_selected %>% 
#  left_join(df) %>% 
#  select(patient_id, everything())


# write_csv(clinical_data_joined, "data/clinical_data_joined.csv")




