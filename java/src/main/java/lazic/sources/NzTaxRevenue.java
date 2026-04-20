package lazic.sources;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import lazic.utils.ingest.DataPoint;
import lazic.utils.ingest.DataSourceBase;
import lazic.utils.ingest.WebHtmlGetter;

public class NzTaxRevenue extends DataSourceBase {
	private final String URL = "https://sdmx.oecd.org/public/rest/data/OECD.CTP.TPS,DSD_REV_COMP_GLOBAL@DF_RSGLOBAL,2.1/NZL...T_1000+T_2000+T_3000+T_4000+T_5000+T_6000+_T..USD+XDC+PT_OTR_SECTOR+PT_B1GQ.A?startPeriod=2000&dimensionAtObservation=AllDimensions";

	@Override
	public String getSourceName() { return "nz_tax_revenue"; }

	@Override
	public Set<DataPoint> getDataPoints() {
		Set<DataPoint> dataPoints = new HashSet<>();
		Gson gson = new Gson();

		try {
			String rawData = WebHtmlGetter.get(URL);
			JsonObject root = gson.fromJson(rawData, JsonObject.class);

			// Extract dimension and attribute mappings
			JsonObject structure = root.getAsJsonObject("structure");
			JsonObject dimensions = structure.getAsJsonObject("dimensions");
			JsonArray observationDims = dimensions.getAsJsonArray("observation");

			// Build lookup maps for dimension values
			Map<String, Map<String, String>> dimensionMaps = buildDimensionMaps(observationDims);

			// Extract observations
			JsonArray dataSets = root.getAsJsonArray("dataSets");
			if (dataSets != null && dataSets.size() > 0) {
				JsonObject dataSet = dataSets.get(0).getAsJsonObject();
				JsonObject observations = dataSet.getAsJsonObject("observations");

				// Process each observation
				for (Map.Entry<String, JsonElement> entry : observations.entrySet()) {
					String key = entry.getKey();
					JsonArray values = entry.getValue().getAsJsonArray();

					// Parse the observation key (format: "0:0:0:3:0:1:0:0")
					String[] indices = key.split(":");

					if (indices.length != 8) {
						continue; // Skip malformed keys
					}

					try {
						// Extract the actual value - handle both string and numeric types
						double value = 0.0;
						boolean hasValue = false;

						// The value array format appears to be: [value_or_empty_string, attr1, attr2, attr3, attr4, attr5]
						// Try to parse the first element as a number
						if (values.size() > 0) {
							JsonElement firstElement = values.get(0);

							if (firstElement.isJsonPrimitive()) {
								if (firstElement.getAsJsonPrimitive().isNumber()) {
									value = firstElement.getAsDouble();
									hasValue = true;
								} else if (firstElement.getAsJsonPrimitive().isString()) {
									String strValue = firstElement.getAsString();
									if (!strValue.isEmpty()) {
										try {
											value = Double.parseDouble(strValue);
											hasValue = true;
										} catch (NumberFormatException e) {
											// Not a number, skip this observation
										}
									}
								}
							}
						}

						// Skip if no valid value found or value is zero
						if (!hasValue || value == 0.0) {
							continue;
						}

						// Extract dimension values using indices
						String sector = getDimensionValue(dimensionMaps, "SECTOR", indices[2]);
						String revenueCategory = getDimensionValue(dimensionMaps, "STANDARD_REVENUE", indices[3]);
						String unitMeasure = getDimensionValue(dimensionMaps, "UNIT_MEASURE", indices[5]);
						String timePeriod = getDimensionValue(dimensionMaps, "TIME_PERIOD", indices[7]);

						// Skip if missing critical dimensions
						if (revenueCategory.isEmpty() || timePeriod.isEmpty()) {
							continue;
						}

						// Parse time period to LocalDateTime
						LocalDateTime dateTime = parseTimePeriod(timePeriod);

						// Create feature name combining the relevant dimensions
						String featureName = String.format("NZ_Tax_%s_%s_%s",
										sanitizeName(revenueCategory),
										sanitizeName(sector),
										sanitizeName(unitMeasure));

						// Create DataPoint (ticker is null for macro data)
						DataPoint dp = new DataPoint(dateTime, null, featureName, value);
						dataPoints.add(dp);

					} catch (Exception e) {
						System.err.println("Error parsing observation key: " + key + " - " + e.getMessage());
					}
				}
			}

			System.out.println("Successfully parsed " + dataPoints.size() + " data points");

		} catch (Exception e) {
			System.err.println("Error fetching or parsing data: " + e.getMessage());
			e.printStackTrace();
		}

		return dataPoints;
	}

	/**
	 * Build lookup maps for dimension values
	 */
	private Map<String, Map<String, String>> buildDimensionMaps(JsonArray observationDims) {
		Map<String, Map<String, String>> dimensionMaps = new HashMap<>();

		for (JsonElement dimElement : observationDims) {
			JsonObject dim = dimElement.getAsJsonObject();
			String dimId = dim.get("id").getAsString();
			JsonArray dimValues = dim.getAsJsonArray("values");

			Map<String, String> valueMap = new HashMap<>();
			for (int i = 0; i < dimValues.size(); i++) {
				JsonObject valueObj = dimValues.get(i).getAsJsonObject();
				String valueId = valueObj.get("id").getAsString();
				String valueName = valueObj.get("name").getAsString();
				valueMap.put(String.valueOf(i), valueName);
			}

			dimensionMaps.put(dimId, valueMap);
		}

		return dimensionMaps;
	}

	/**
	 * Get dimension value by dimension name and index
	 */
	private String getDimensionValue(Map<String, Map<String, String>> dimensionMaps,
																	 String dimensionName, String index) {
		Map<String, String> valueMap = dimensionMaps.get(dimensionName);
		if (valueMap == null) {
			return "";
		}
		return valueMap.getOrDefault(index, "");
	}

	/**
	 * Parse time period string to LocalDateTime
	 */
	private LocalDateTime parseTimePeriod(String timePeriod) {
		try {
			// Time period is in format "2000", "2001", etc.
			int year = Integer.parseInt(timePeriod);
			return LocalDateTime.of(year, 1, 1, 0, 0); // Set to Jan 1st of the year
		} catch (NumberFormatException e) {
			return LocalDateTime.now(); // Fallback to current time
		}
	}

	/**
	 * Sanitize dimension names for feature naming
	 */
	private String sanitizeName(String name) {
		return name.replaceAll("[^a-zA-Z0-9_]", "_")
						.replaceAll("_+", "_")
						.replaceAll("^_|_$", "");
	}
}