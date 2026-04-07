import React,{useState} from "react";
function FileUpload({onUploadSuccess,selectedModels,usePCA}) {
    const [file, setFile] = useState(null);

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");



    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
        setError("");
    };

    const handleUpload = async () => {
        console.log("Current models in FileUpload:", selectedModels);
        if (!file) {
            setError("Please select file first!");
            return;
        }

        if (!selectedModels || selectedModels.length === 0) {
            setError("Please select at least one model!");
            return;
        }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("models_names", JSON.stringify(selectedModels));
    formData.append("use_pca",usePCA?"true":"false");

    try {
        setLoading(true);
        const response = await fetch("http://127.0.0.1:8000/run", {
            method: "POST",
            body: formData,
        });


        if (!response.ok) {
            // 💡 specific 422 errors
            if (response.status === 422) {
                const errorDetail = await response.json();
                console.error("FastAPI Validation Error:", errorDetail);
                throw new Error(`Parameter Error: ${JSON.stringify(errorDetail.detail)}`);
            }
            throw new Error("Upload failed");
        }
        const data = await response.json();
        if (onUploadSuccess) {
            // if data.results exists，ors data
            const finalData = data.results ? data.results : data;
            onUploadSuccess(finalData);
        }


    } catch (err) {
        setError(err.message);
    } finally {
        setLoading(false);
    }
};

return (
        <div style={{border: "1px solid #ccc", padding: "16px"}}>
            <h3>Upload File</h3>
            <input type="file" onChange={handleFileChange}></input>
            {file && <p>Selected:{file.name}</p>}
            <button onClick={handleUpload} disabled={loading}>
                {loading ? "Uploading ..." : "Upload"}</button>
            {error && <p style={{ color: "red" }}>{error}</p>}
        </div>
    );
}
export default  FileUpload;