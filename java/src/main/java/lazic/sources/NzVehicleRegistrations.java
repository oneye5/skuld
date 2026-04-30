package lazic.sources;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import lazic.utils.ingest.Cadence;
import lazic.utils.ingest.DataPoint;
import lazic.utils.ingest.DataSourceBase;
import lazic.utils.ingest.ReleaseDate;
import lazic.utils.ingest.ReleaseLag;
import lazic.utils.ingest.WebHtmlGetter;

public class NzVehicleRegistrations extends DataSourceBase {
	// The URL provided in the snippet
	private final String URL = "https://sdmx.oecd.org/public/rest/data/OECD.ITF,DSD_ST@DF_STREG,1.0/NZL.M...ROAD...";

	// NZTA / OECD-ITF monthly vehicle registration stats: published ~25 days after month-end.
	private static final ReleaseLag RELEASE_LAG = ReleaseLag.of(25);

	/**
	 * Returns a set of DataPoint's.
	 * Ticker is null as this is macroeconomic data (New Zealand Vehicle Registrations).
	 */
	@Override
	public String getSourceName() { return "nz_vehicle_registrations"; }

	@Override
	public Set<DataPoint> getDataPoints() {
		Set<DataPoint> dataPoints = new HashSet<>();

		// 1. Fetch the raw JSON
		String rawData = WebHtmlGetter.get(URL);
		if (rawData == null || rawData.isEmpty()) {
			System.err.println("No data received from URL: " + URL);
			return dataPoints;
		}

		try {
			// 2. Parse the JSON tree
			JsonObject root = JsonParser.parseString(rawData).getAsJsonObject();
			JsonObject structure = root.getAsJsonObject("structure");
			JsonArray dataSets = root.getAsJsonArray("dataSets");

			if (dataSets.size() == 0) return dataPoints;

			// 3. Build the Time Index Map (Index -> LocalDateTime)
			// In SDMX-JSON, the "observation" dimension defines the time periods.
			Map<Integer, LocalDateTime> timeIndexMap = buildTimeIndexMap(structure);

			// 4. Build the Vehicle Type Map (Dimension Index -> Name)
			// Based on the file provided, VEHICLE_TYPE is at index 6 of the dimensions.series list.
			Map<Integer, String> vehicleTypeMap = buildVehicleTypeMap(structure);

			// 5. Iterate through the Series in the DataSet
			JsonObject seriesMap = dataSets.get(0).getAsJsonObject().getAsJsonObject("series");

			for (Map.Entry<String, JsonElement> entry : seriesMap.entrySet()) {
				String dimensionString = entry.getKey(); // e.g., "0:0:0:0:0:0:1:0"
				JsonObject seriesObj = entry.getValue().getAsJsonObject();

				// Determine the feature name based on the dimension key
				String featureName = getFeatureName(dimensionString, vehicleTypeMap);
				if (featureName == null) continue;

				// 6. Iterate through Observations for this Series
				JsonObject observations = seriesObj.getAsJsonObject("observations");

				for (Map.Entry<String, JsonElement> obsEntry : observations.entrySet()) {
					try {
						int timeIndex = Integer.parseInt(obsEntry.getKey());

						// The value is strictly an array in the source: "0":[3929]
						JsonArray valueArr = obsEntry.getValue().getAsJsonArray();
						if (valueArr.size() > 0 && !valueArr.get(0).isJsonNull()) {
							Double value = valueArr.get(0).getAsDouble();
							LocalDateTime date = timeIndexMap.get(timeIndex);

							if (date != null) {
								// Create the DataPoint
								// Ticker is null for macro data
								dataPoints.add(new DataPoint(date, null, featureName, value));
							}
						}
					} catch (NumberFormatException | IndexOutOfBoundsException e) {
						// Skip malformed observations
					}
				}
			}

		} catch (Exception e) {
			e.printStackTrace();
			System.err.println("Error parsing SDMX JSON: " + e.getMessage());
		}

		return dataPoints;
	}

	/**
	 * Parses structure.dimensions.observation to map indices (0, 1, 2...) to Dates.
	 */
	private Map<Integer, LocalDateTime> buildTimeIndexMap(JsonObject structure) {
		Map<Integer, LocalDateTime> map = new HashMap<>();
		try {
			JsonObject dimensions = structure.getAsJsonObject("dimensions");
			JsonArray observationDims = dimensions.getAsJsonArray("observation");

			// usually the first element in 'observation' holds the time periods
			JsonObject timeDim = observationDims.get(0).getAsJsonObject();
			JsonArray values = timeDim.getAsJsonArray("values");

			DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");

			for (int i = 0; i < values.size(); i++) {
				JsonObject val = values.get(i).getAsJsonObject();
				String dateStr = val.get("id").getAsString(); // e.g., "2019-02"

				// Append -01 to make it a valid ISO date for parsing
				LocalDate ld = LocalDate.parse(dateStr + "-01", formatter);
				LocalDateTime periodStart = ld.atStartOfDay();
				map.put(i, ReleaseDate.applyLag(periodStart, Cadence.MONTHLY, RELEASE_LAG));
			}
		} catch (Exception e) {
			System.err.println("Error building time index map: " + e.getMessage());
		}
		return map;
	}

	/**
	 * Parses structure.dimensions.series to map indices for VEHICLE_TYPE.
	 * In the provided data, VEHICLE_TYPE is at keyPosition 6.
	 */
	private Map<Integer, String> buildVehicleTypeMap(JsonObject structure) {
		Map<Integer, String> map = new HashMap<>();
		try {
			JsonObject dimensions = structure.getAsJsonObject("dimensions");
			JsonArray seriesDims = dimensions.getAsJsonArray("series");

			// Find the dimension with id "VEHICLE_TYPE"
			for (JsonElement dim : seriesDims) {
				JsonObject dimObj = dim.getAsJsonObject();
				if ("VEHICLE_TYPE".equals(dimObj.get("id").getAsString())) {
					JsonArray values = dimObj.getAsJsonArray("values");
					for (int i = 0; i < values.size(); i++) {
						JsonObject val = values.get(i).getAsJsonObject();
						// e.g., id="GV", name="Goods road motor vehicles"
						// e.g., id="CARS", name="Passenger cars"
						map.put(i, val.get("name").getAsString());
					}
					break;
				}
			}
		} catch (Exception e) {
			System.err.println("Error building vehicle type map: " + e.getMessage());
		}
		return map;
	}

	/**
	 * Decodes the dimension string (e.g., "0:0:0:0:0:0:1:0") to find the vehicle type.
	 */
	private String getFeatureName(String dimensionString, Map<Integer, String> vehicleTypeMap) {
		// The dimensions are separated by ":"
		// Based on "structure", VEHICLE_TYPE is at index 6.
		// 0: REF_AREA
		// 1: FREQ
		// 2: MEASURE
		// 3: UNIT_MEASURE
		// 4: TRANSPORT_MODE
		// 5: GEO_COVERAGE
		// 6: VEHICLE_TYPE <--- Target
		// 7: FUEL

		String[] parts = dimensionString.split(":");
		if (parts.length <= 6) return "Unknown Vehicle Data";

		try {
			int typeIndex = Integer.parseInt(parts[6]);
			String typeName = vehicleTypeMap.get(typeIndex);

			if (typeName != null) {
				return "New Registrations - " + typeName;
			}
		} catch (NumberFormatException e) {
			return "Unknown Vehicle Data";
		}

		return "Unknown Vehicle Data";
	}
}