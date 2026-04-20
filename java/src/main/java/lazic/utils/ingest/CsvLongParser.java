package lazic.utils.ingest;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.file.Path;
import java.time.ZoneId;
import java.util.LinkedHashMap;
import java.util.Map;

public class CsvLongParser {

	/**
	 * Saves all DataPoint entries to long-format CSV:
	 *
	 * timestamp,ticker,feature,value,src
	 *
	 * Also writes a source_legend.csv alongside the main file.
	 */
	public static boolean saveCsv(String path) {
		var data = IngestManager.INSTANCE.data;

		if (data.isEmpty()) {
			return false;
		}

		// Build source name → integer mapping
		Map<String, Integer> sourceLegend = new LinkedHashMap<>();
		for (DataPoint dp : data) {
			String src = dp.getSource() != null ? dp.getSource() : "";
			sourceLegend.computeIfAbsent(src, k -> sourceLegend.size());
		}

		// Write the legend file alongside the main CSV
		Path legendPath = Path.of(path).getParent().resolve("source_legend.csv");
		writeSourceLegend(legendPath.toString(), sourceLegend);

		try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
			bw.write("timestamp,ticker,feature,value,src");
			bw.newLine();

			for (DataPoint dp : data) {
				long ts = (dp.getTimestamp() != null)
								? dp.getTimestamp()
								.atZone(ZoneId.of("UTC"))
								.toInstant()
								.toEpochMilli()
								: -1;

				String ticker = dp.getTicker() != null ? dp.getTicker() : "";
				String feature = normalizeFeatureName(dp.getFeatureName());
				String value = formatValue(dp.getValue());
				int src = sourceLegend.getOrDefault(
								dp.getSource() != null ? dp.getSource() : "", 0);

				bw.write(ts + "," + csvField(ticker) + "," + csvField(feature) + "," + value + "," + src);
				bw.newLine();
			}

			return true;

		} catch (IOException e) {
			e.printStackTrace();
			return false;
		}
	}

	/**
	 * Normalize feature names to consistent snake_case.
	 */
	static String normalizeFeatureName(String name) {
		if (name == null || name.isEmpty()) return "";
		// Insert underscore before uppercase letters (camelCase → camel_Case)
		name = name.replaceAll("([a-z0-9])([A-Z])", "$1_$2");
		// Replace spaces, dashes, commas, parentheses, colons, semicolons with underscores
		name = name.replaceAll("[\\s\\-,():;]+", "_");
		// Collapse multiple underscores
		name = name.replaceAll("_+", "_");
		// Strip leading/trailing underscores
		name = name.replaceAll("^_|_$", "");
		// Lowercase
		return name.toLowerCase();
	}

	/**
	 * Format value without scientific notation.
	 */
	private static String formatValue(Double value) {
		if (value == null) return "";
		return BigDecimal.valueOf(value).stripTrailingZeros().toPlainString();
	}

	/**
	 * RFC 4180 CSV field quoting.
	 */
	private static String csvField(String field) {
		if (field == null) return "";
		if (field.contains(",") || field.contains("\"") || field.contains("\n") || field.contains("\r")) {
			return "\"" + field.replace("\"", "\"\"") + "\"";
		}
		return field;
	}

	/**
	 * Write source legend mapping file.
	 */
	private static void writeSourceLegend(String path, Map<String, Integer> sourceLegend) {
		try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
			bw.write("id,name");
			bw.newLine();
			sourceLegend.entrySet().stream()
							.sorted(Map.Entry.comparingByValue())
							.forEach(entry -> {
								try {
									bw.write(entry.getValue() + "," + entry.getKey());
									bw.newLine();
								} catch (IOException e) {
									throw new RuntimeException(e);
								}
							});
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
