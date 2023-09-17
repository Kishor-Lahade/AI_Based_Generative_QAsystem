document
  .getElementById("questionForm")
  .addEventListener("answer", function (event) {
    event.preventDefault();

    const emailText = document.getElementById("questionText").value;

    fetch(`http://127.0.0.1:8000/answer/${encodeURIComponent(emailText)}`)
      .then((response) => response.json())
      .then((data) => {
        const responseDiv = document.getElementById("responseDiv");
        responseDiv.innerHTML = `<strong>Answer:</strong> ${data.anser}`;
        console.log(data);
      })
      .catch((error) => {
        const responseDiv = document.getElementById("responseDiv");
        responseDiv.innerHTML = `<strong>Error:</strong> An error occurred while fetching the data.`;
        console.error("Error:", error);
      });
  });
