document.getElementById("employeeForm").addEventListener("submit", function (e) {
  e.preventDefault();

  const formData = new FormData(e.target);
  const data = Object.fromEntries(formData.entries());

  function describeProductivity(score) {
    if (score >= 85) return "висока";
    if (score >= 70) return "середня";
    return "низька";
  }
  
  function describeRisk(risk) {
    if (risk >= 0.7) return "високий ризик";
    if (risk >= 0.4) return "середній ризик";
    return "низький ризик";
  }
  

  fetch("/analyze", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(data)
  })
    .then(res => res.json())
    .then(result => {
      console.log("Відповідь API:", result); // для налагодження

      if (!result || typeof result.predicted_productivity === "undefined") {
        alert("Помилка під час аналізу. Перевірте дані або спробуйте пізніше.");
        return;
      }

      // Показ результатів
      
      document.getElementById("result").classList.remove("d-none");
      document.getElementById("productivityScore").innerText =
        `${result.predicted_productivity}/100 — ${describeProductivity(result.predicted_productivity)}`;

      document.getElementById("attritionRisk").innerText =
        `${Math.round(result.attrition_risk * 100)}% — ${describeRisk(result.attrition_risk)}`;
      document.getElementById("explanation").innerText = result.explanation;
      
      
      /*document.getElementById("result").classList.remove("d-none");
      document.getElementById("productivityScore").innerText = result.predicted_productivity;
      document.getElementById("attritionRisk").innerText = result.attrition_risk;
      //document.getElementById("cluster").innerText = result.cluster;
      document.getElementById("explanation").innerText = result.explanation;
      */
      // Список рекомендацій
      const recList = document.getElementById("recommendations");
      recList.innerHTML = "";
      result.recommendations.forEach(r => {
        const li = document.createElement("li");
        li.className = "list-group-item";
        li.innerText = r;
        recList.appendChild(li);
      });

      // === Побудова графіка порівняння ===
      const canvas = document.getElementById("chart");
      const ctx = canvas.getContext("2d");

      if (window.chart instanceof Chart && typeof window.chart.destroy === "function") {
        window.chart.destroy();
      }

      window.chart = new Chart(ctx, {
        type: "bar",
        data: {
          labels: ["Продуктивність", "Добробут", "Вік", "Дохід"],
          datasets: [
            {
              label: "Поточний працівник",
              data: [
                parseFloat(result.predicted_productivity),
                parseFloat(data.WellBeingScore),
                parseFloat(data.Age),
                parseFloat(data.MonthlyIncome)
              ],
              backgroundColor: "rgba(54, 162, 235, 0.6)"
            },
            {
              label: "Середнє по компанії",
              data: [70, 65, 35, 5000], // Можна замінити на динамічні значення з API
              backgroundColor: "rgba(255, 206, 86, 0.5)"
            }
          ]
        },
        options: {
          responsive: true,
          scales: {
            y: {
              beginAtZero: true
            }
          },
          plugins: {
            title: {
              display: true,
              text: "Порівняння з середніми значеннями по компанії"
            }
          }
        }
      });
    })
    .catch(err => {
      console.error("Помилка при запиті:", err);
      alert("Сталася помилка. Перевірте підключення або введені дані.");
    });
});


/*document.getElementById("employeeForm").addEventListener("submit", function (e) {
  e.preventDefault();

  const formData = new FormData(e.target);
  const data = Object.fromEntries(formData.entries());

  fetch("/analyze", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(data)
  })
    .then(res => res.json())
    .then(result => {
      console.log("Відповідь API:", result); // для перевірки

      if (!result || !result.predicted_productivity) {
        alert("Помилка під час аналізу. Перевірте дані або спробуйте пізніше.");
        return;
      }

      document.getElementById("result").classList.remove("d-none");
      document.getElementById("employeeName").innerText = `Результати для працівника ${result.employee_id}`;
      document.getElementById("productivityScore").innerText = `${result.predicted_productivity}`;
      document.getElementById("attritionRisk").innerText = `${result.attrition_risk}`;
      document.getElementById("cluster").innerText = `${result.cluster}`;
      document.getElementById("explanation").innerText = result.explanation;

      const recList = document.getElementById("recommendations");
      recList.innerHTML = "";
      result.recommendations.forEach(r => {
        const li = document.createElement("li");
        li.className = "list-group-item";
        li.innerText = r;
        recList.appendChild(li);
      });

      // === Побудова графіка ===
      const canvas = document.getElementById("chart");
      const ctx = canvas.getContext("2d");

      if (window.chart instanceof Chart && typeof window.chart.destroy === "function") {
        window.chart.destroy();
      }
      

      window.chart = new Chart(ctx, {
        type: "bar",
        data: {
          labels: ["Продуктивність", "Добробут", "Вік", "Дохід"],
          datasets: [
            {
              label: "Працівник",
              data: [
                parseFloat(result.predicted_productivity),
                parseFloat(data.WellBeingScore),
                parseFloat(data.Age),
                parseFloat(data.MonthlyIncome)
              ],
              backgroundColor: "rgba(54, 162, 235, 0.6)"
            },
            {
              label: "Середнє по компанії",
              data: [70, 65, 35, 5000], // Можна замінити на реальні дані з бекенда
              backgroundColor: "rgba(255, 206, 86, 0.5)"
            }
          ]
        },
        options: {
          responsive: true,
          scales: {
            y: {
              beginAtZero: true
            }
          },
          plugins: {
            title: {
              display: true,
              text: "Порівняння працівника з середніми значеннями"
            }
          }
        }
      });
    })
    .catch(err => {
      console.error("Помилка під час виконання:", err);
      alert("Сталася помилка. Перевірте дані або спробуйте пізніше.");
    });
});


*/