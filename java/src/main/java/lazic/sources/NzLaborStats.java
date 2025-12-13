package lazic.sources;

import java.time.LocalDateTime;
import java.util.Set;

import com.google.gson.Gson;

import lazic.utils.ingest.DataPoint;
import lazic.utils.ingest.DataSourceBase;
import lazic.utils.ingest.WebHtmlGetter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.io.StringReader;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;

public class NzLaborStats extends DataSourceBase {
	private final String URL = "https://sdmx.oecd.org/public/rest/data/OECD.CFE.EDS,DSD_REG_LABOUR@DF_LAB,2.0/A..NZL..POP+LF+EMP+UNE+LF_RATE+UNE_RATE+LF_RATE_SEXDIF+EMP_RATIO_SEXDIF+UNE_LT+UNE_LT_RATE+UNE_RATE_SEXDIF+EMP_RATIO.Y15T24+Y_GT15+Y15T64.M+F+_T.?startPeriod=1996&dimensionAtObservation=AllDimensions";

	/**
	 * Returns a set of DataPoint's. Ticker is null if the datapoint does not pertain to a particular ticker, such as macroeconomic data for example
	 * There are multiple DataPoint's in a time-series feature, and there may be multiple features returned overall.
	 */
	@Override
	public Set<DataPoint> getDataPoints() {

		// fetch raw XML from remote
		String rawData = WebHtmlGetter.get(URL);

		if (rawData == null || rawData.isBlank()) {
			System.err.println("NzLaborStats: no data retrieved from source.");
			return java.util.Collections.emptySet();
		}

		// Optional: save raw response for debugging/analysis when system property set
		try {
			if ("true".equalsIgnoreCase(System.getProperty("skuld.saveRaw", "false"))) {
				Path out = Paths.get("target", "data", "NzLaborStats_raw.txt");
				Files.createDirectories(out.getParent());
				Files.writeString(out, rawData, StandardCharsets.UTF_8, StandardOpenOption.CREATE,
						StandardOpenOption.TRUNCATE_EXISTING);
			}
		} catch (IOException e) {
			System.err.println("NzLaborStats: failed to write raw data: " + e.getMessage());
		}

		Set<DataPoint> parsed = parseXmlData(rawData);

		// Optional: write parsed CSV for debugging when system property set
		try {
			if ("true".equalsIgnoreCase(System.getProperty("skuld.saveParsed", "false"))) {
				Path parsedOut = Paths.get("target", "data", "NzLaborStats_parsed.csv");
				Files.createDirectories(parsedOut.getParent());
				try (java.io.BufferedWriter bw = Files.newBufferedWriter(parsedOut, StandardCharsets.UTF_8,
						StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
					bw.write("timestamp,ticker,feature,value");
					bw.newLine();
					for (DataPoint dp : parsed) {
						String ts = dp.getTimestamp() == null ? "" : dp.getTimestamp().toString();
						String tic = dp.getTicker() == null ? "" : dp.getTicker();
						String feat = dp.getFeatureName() == null ? "" : dp.getFeatureName();
						String val = dp.getValue() == null ? "" : dp.getValue().toString();
						bw.write(ts + "," + tic + "," + feat + "," + val);
						bw.newLine();
					}
				}
			}
		} catch (IOException e) {
			System.err.println("NzLaborStats: failed to write parsed CSV: " + e.getMessage());
		}

		return parsed;
	}

	/**
	 * Parses SDMX-ML generic XML and extracts basic observations.
	 * For each <generic:Obs> block we extract TIME_PERIOD, MEASURE (if present), AGE, SEX and ObsValue.
	 */
	private Set<DataPoint> parseXmlData(String rawData) {
		var dataPoints = new java.util.HashSet<DataPoint>();

		if (rawData == null || rawData.isBlank()) return dataPoints;

		try {
			DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
			dbf.setNamespaceAware(false);
			DocumentBuilder db = dbf.newDocumentBuilder();
			InputSource is = new InputSource(new StringReader(rawData));
			Document doc = db.parse(is);

			NodeList obsNodes = doc.getElementsByTagName("generic:Obs");
			if (obsNodes == null || obsNodes.getLength() == 0) {
				obsNodes = doc.getElementsByTagName("Obs");
			}

			for (int i = 0; i < obsNodes.getLength(); i++) {
				Node obsNode = obsNodes.item(i);
				if (obsNode == null || obsNode.getNodeType() != Node.ELEMENT_NODE) continue;

				String time = null;
				String measure = null;
				String age = null;
				String sex = null;
				String obsVal = null;
				int unitMult = 0;

				// iterate children of <generic:Obs>
				NodeList obsChildren = obsNode.getChildNodes();
				for (int j = 0; j < obsChildren.getLength(); j++) {
					Node c = obsChildren.item(j);
					if (c.getNodeType() != Node.ELEMENT_NODE) continue;
					String nodeName = c.getNodeName();

					if (nodeName.endsWith("ObsKey") || nodeName.equals("generic:ObsKey") || nodeName.equals("ObsKey")) {
						NodeList keyVals = c.getChildNodes();
						for (int k = 0; k < keyVals.getLength(); k++) {
							Node kv = keyVals.item(k);
							if (kv.getNodeType() != Node.ELEMENT_NODE) continue;
							if (!kv.getNodeName().endsWith("Value")) continue;
							Element e = (Element) kv;
							String id = e.getAttribute("id");
							String val = e.getAttribute("value");
							if (id == null) continue;
							switch (id) {
								case "TIME_PERIOD": time = val; break;
								case "MEASURE": measure = val; break;
								case "AGE": age = val; break;
								case "SEX": sex = val; break;
								default: break;
							}
						}

					} else if (nodeName.endsWith("ObsValue") || nodeName.equals("generic:ObsValue") || nodeName.equals("ObsValue")) {
						Element e = (Element) c;
						obsVal = e.getAttribute("value");

					} else if (nodeName.endsWith("Attributes") || nodeName.equals("generic:Attributes") || nodeName.equals("Attributes")) {
						NodeList attrs = c.getChildNodes();
						for (int k = 0; k < attrs.getLength(); k++) {
							Node av = attrs.item(k);
							if (av.getNodeType() != Node.ELEMENT_NODE) continue;
							if (!av.getNodeName().endsWith("Value")) continue;
							Element ae = (Element) av;
							String id = ae.getAttribute("id");
							String val = ae.getAttribute("value");
							if ("UNIT_MULT".equals(id) && val != null && !val.isEmpty()) {
								try { unitMult = Integer.parseInt(val); } catch (NumberFormatException _ex) { unitMult = 0; }
							}
						}
					}
				}

				if (time == null) continue;

				try {
					LocalDateTime timestamp;
					if (time.contains("-Q")) {
						String[] parts = time.split("-Q");
						int year = Integer.parseInt(parts[0]);
						int q = Integer.parseInt(parts[1]);
						int month = switch (q) { case 1 -> 1; case 2 -> 4; case 3 -> 7; case 4 -> 10; default -> 1; };
						timestamp = LocalDateTime.of(year, month, 1, 0, 0);
					} else if (time.matches("\\\\d{4}-\\\\d{2}-\\\\d{2}T.*")) {
						timestamp = LocalDateTime.parse(time);
					} else if (time.matches("\\\\d{4}-\\\\d{2}")) {
						timestamp = LocalDateTime.parse(time + "-01T00:00:00");
					} else {
						int year = Integer.parseInt(time);
						timestamp = LocalDateTime.of(year, 1, 1, 0, 0);
					}

					String featureName = "NZL_LaborStats";
					if (measure != null && !measure.isEmpty()) featureName += "_" + measure;
					if (age != null && !age.isEmpty()) featureName += "_" + age;
					if (sex != null && !sex.isEmpty()) featureName += "_" + sex;

					double value = Double.NaN;
					if (obsVal != null && !obsVal.isEmpty()) {
						try { value = Double.parseDouble(obsVal); } catch (NumberFormatException nfe) { value = Double.NaN; }
						if (!Double.isNaN(value) && unitMult != 0) {
							value = value * Math.pow(10, unitMult);
						}
					}

					dataPoints.add(new DataPoint(timestamp, null, featureName, value));

				} catch (Exception ex) {
					// skip malformed
				}
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		return dataPoints;
	}
}
