package lazic.sources;

import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
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

public class GlobalAquacultureProduction extends DataSourceBase {
	private final String URL = "https://sdmx.oecd.org/public/rest/data/OECD.TAD.ARP,DSD_FISH_PROD@DF_FISH_AQUA,1.0/.A.._T.T?startPeriod=2000&dimensionAtObservation=AllDimensions";

	/**
	 * Returns a set of DataPoint's. Ticker is null if the datapoint does not pertain to a particular ticker,
	 * such as macroeconomic data for example. There are multiple DataPoint's in a time-series feature,
	 * and there may be multiple features returned overall.
	 */
	@Override
	public String getSourceName() { return "global_aquaculture_production"; }

	@Override
	public Set<DataPoint> getDataPoints() {
		String[] tickers = lazic.sources.config.Tickers.TICKERS;
		Gson gson = new Gson();
		String rawData = WebHtmlGetter.get(URL);

		try (FileWriter writer = new FileWriter("sample_data.txt")) {
			writer.write(rawData);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

		return parseAquacultureData(rawData, gson);
	}

	private Set<DataPoint> parseAquacultureData(String rawData, Gson gson) {
		Set<DataPoint> dataPoints = new HashSet<>();

		try {
			JsonObject root = gson.fromJson(rawData, JsonObject.class);
			JsonArray dataSets = root.getAsJsonArray("dataSets");

			if (dataSets == null || dataSets.size() == 0) {
				return dataPoints;
			}

			JsonObject dataSet = dataSets.get(0).getAsJsonObject();
			JsonObject observations = dataSet.getAsJsonObject("observations");

			// Get dimension metadata from structure
			JsonObject structure = root.getAsJsonObject("structure");
			JsonObject dimensions = structure.getAsJsonObject("dimensions");
			JsonArray observationDims = dimensions.getAsJsonArray("observation");

			// Build lookup maps for dimension values
			Map<Integer, JsonArray> dimensionValues = new java.util.HashMap<>();
			for (int i = 0; i < observationDims.size(); i++) {
				JsonObject dim = observationDims.get(i).getAsJsonObject();
				JsonArray values = dim.getAsJsonArray("values");
				dimensionValues.put(i, values);
			}

			// Parse each observation
			for (Map.Entry<String, JsonElement> entry : observations.entrySet()) {
				String key = entry.getKey();
				JsonArray obsValue = entry.getValue().getAsJsonArray();

				// Key format: "REF_AREA:FREQ:MEASURE:SPECIES:UNIT_MEASURE:TIME_PERIOD"
				// Example: "0:0:0:0:0:0" maps to dimensions
				String[] indices = key.split(":");

				if (indices.length < 6) continue;

				// Extract dimension values
				int refAreaIdx = Integer.parseInt(indices[0]);
				int timePeriodIdx = Integer.parseInt(indices[5]);

				// Get country name
				JsonArray refAreaValues = dimensionValues.get(0);
				String countryId = refAreaValues.get(refAreaIdx).getAsJsonObject().get("id").getAsString();
				String countryName = refAreaValues.get(refAreaIdx).getAsJsonObject().get("name").getAsString();

				// Get time period (year)
				JsonArray timePeriodValues = dimensionValues.get(5);
				String year = timePeriodValues.get(timePeriodIdx).getAsJsonObject().get("id").getAsString();

				// Get observation value (first element in array)
				if (obsValue.size() == 0 || obsValue.get(0).isJsonNull()) continue;
				double value = obsValue.get(0).getAsDouble();

				// Create timestamp from year
				LocalDateTime timestamp = LocalDateTime.of(Integer.parseInt(year), 1, 1, 0, 0);

				// Feature name includes country for clarity
				String featureName = "aquaculture_production_tonnes_" + countryId.toLowerCase();

				// Ticker is null for macroeconomic/country-level data
				DataPoint dp = new DataPoint(timestamp, null, featureName, value);
				dataPoints.add(dp);
			}

		} catch (Exception e) {
			System.err.println("Error parsing aquaculture data: " + e.getMessage());
			e.printStackTrace();
		}

		return dataPoints;
	}
}