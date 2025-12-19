package lazic.sources;

import java.io.FileWriter;
import java.time.LocalDateTime;
import java.util.Set;
import java.util.HashSet;
import java.util.Map;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;

import lazic.utils.ingest.DataPoint;
import lazic.utils.ingest.DataSourceBase;
import lazic.utils.ingest.WebHtmlGetter;
import java.io.IOException;

public class NzLaborStats extends DataSourceBase {
	private final String URL = "https://sdmx.oecd.org/public/rest/data/OECD.CFE.EDS,DSD_REG_LABOUR@DF_LAB,2.0/A..NZL..POP+LF+EMP+UNE+LF_RATE+UNE_RATE+LF_RATE_SEXDIF+EMP_RATIO_SEXDIF+UNE_LT+UNE_LT_RATE+UNE_RATE_SEXDIF+EMP_RATIO.Y15T24+Y_GT15+Y15T64.M+F+_T.?startPeriod=1996&dimensionAtObservation=AllDimensions";

	/**
	 * Returns a set of DataPoint's. Ticker is null if the datapoint does not pertain to a particular ticker, such as macroeconomic data for example
	 * There are multiple DataPoint's in a time-series feature, and there may be multiple features returned overall.
	 */
	@Override
	public Set<DataPoint> getDataPoints() {
		Set<DataPoint> dataPoints = new HashSet<>();
		Gson gson = new Gson();
		String rawData = WebHtmlGetter.get(URL);

		try {
			JsonObject root = gson.fromJson(rawData, JsonObject.class);
			JsonArray dataSets = root.getAsJsonArray("dataSets");

			if (dataSets == null || dataSets.size() == 0) {
				return dataPoints;
			}

			JsonObject dataSet = dataSets.get(0).getAsJsonObject();
			JsonObject observations = dataSet.getAsJsonObject("observations");

			// Get the prepared date from header
			JsonObject header = root.getAsJsonObject("header");
			String preparedDate = header.get("prepared").getAsString();
			LocalDateTime timestamp = LocalDateTime.parse(preparedDate.substring(0, 19));

			// Parse each observation
			for (Map.Entry<String, JsonElement> entry : observations.entrySet()) {
				String key = entry.getKey();
				JsonArray values = entry.getValue().getAsJsonArray();

				// The first element in the array is the observation value
				if (values.size() > 0 && !values.get(0).isJsonNull()) {
					double value = values.get(0).getAsDouble();

					// Parse the dimension key (format: "dim1:dim2:dim3:...")
					String[] dimensions = key.split(":");

					// Build a feature name from the dimensions
					// Based on SDMX structure: FREQ:MEASURE:REF_AREA:SECTOR:MEASURE_TYPE:AGE:SEX:UNIT_MEASURE:TIME_PERIOD
					String featureName = buildFeatureName(dimensions);

					// Ticker is null for macroeconomic data
					DataPoint dp = new DataPoint(timestamp, null, featureName, value);
					dataPoints.add(dp);
				}
			}

			// Optional: save for debugging
			try (FileWriter writer = new FileWriter("sample_data.txt")) {
				writer.write(rawData);
			} catch (IOException e) {
				System.err.println("Warning: Could not write debug file: " + e.getMessage());
			}

		} catch (Exception e) {
			System.err.println("Error parsing SDMX-JSON data: " + e.getMessage());
			e.printStackTrace();
		}

		return dataPoints;
	}

	private String buildFeatureName(String[] dimensions) {
		// Map dimension codes to readable names
		// Common SDMX dimensions for labor statistics
		StringBuilder name = new StringBuilder("NZ_Labor_");

		// Add indicator type (dimension 4 typically contains measure type)
		if (dimensions.length > 4) {
			name.append(getMeasureLabel(dimensions[4])).append("_");
		}

		// Add age group (dimension 5)
		if (dimensions.length > 5) {
			name.append(getAgeLabel(dimensions[5])).append("_");
		}

		// Add sex (dimension 6)
		if (dimensions.length > 6) {
			name.append(getSexLabel(dimensions[6]));
		}

		// Add time period (dimension 8)
		if (dimensions.length > 8) {
			name.append("_").append(dimensions[8]);
		}

		return name.toString().replaceAll("_+$", ""); // Remove trailing underscores
	}

	private String getMeasureLabel(String code) {
		switch (code) {
			case "0": return "Population";
			case "1": return "LaborForce";
			case "2": return "Employment";
			case "3": return "Unemployment";
			case "4": return "UnemploymentNotWorked";
			case "5": return "LFParticipationRate";
			case "6": return "EmploymentPopRatio";
			case "7": return "UnemploymentRate";
			case "8": return "UnemploymentRateNotWorked";
			case "9": return "YouthUnemploymentRatio";
			case "10": return "LFParticipationGap";
			case "11": return "EmploymentPopRatioGap";
			default: return "Measure" + code;
		}
	}

	private String getAgeLabel(String code) {
		switch (code) {
			case "0": return "Age15to64";
			case "1": return "Age15to24";
			case "2": return "Age15Plus";
			default: return "Age" + code;
		}
	}

	private String getSexLabel(String code) {
		switch (code) {
			case "0": return "Total";
			case "1": return "Male";
			case "2": return "Female";
			default: return "Sex" + code;
		}
	}
}