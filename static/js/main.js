document.addEventListener("DOMContentLoaded", () => {
  const reset = document.getElementById("resetForm");
  const form = document.getElementById("predictionForm");
  const loader = document.getElementById("predictionLoader");
  const status = document.getElementById("loadingStatus");
  const bar = document.getElementById("loadingBar");
  const predictButton = document.getElementById("predictButton");

  if (reset && form) {
    reset.addEventListener("click", () => {
      const demo = {
        location: "thane",
        locality_hint: "pokhran road",
        society: "dosti vihar",
        property_type: "apartment",
        bhk: "2",
        area_sqft: "1200",
        bathrooms: "2",
        balconies: "1",
        current_floor: "5",
        total_floors: "14",
        transaction: "resale",
        furnishing: "semi-furnished",
        facing: "east",
        parking: "1",
        ownership: "freehold",
        overlooking: "garden/park"
      };
      Object.entries(demo).forEach(([key, value]) => {
        const field = form.elements.namedItem(key);
        if (field) field.value = value;
      });
      form.classList.remove("was-validated");
    });
  }

  if (form && loader) {
    const statuses = [
      "Validating property details",
      "Engineering location and property features",
      "Running the trained LightGBM model",
      "Preparing your estimated house price"
    ];

    form.addEventListener("submit", (event) => {
      if (!form.checkValidity()) {
        event.preventDefault();
        event.stopPropagation();
        form.classList.add("was-validated");
        const firstInvalid = form.querySelector(":invalid");
        if (firstInvalid) firstInvalid.focus();
        return;
      }

      // Keep the polished loading experience visible before navigation.
      event.preventDefault();
      loader.classList.add("show");
      loader.setAttribute("aria-hidden", "false");
      document.body.classList.add("loader-open");
      if (predictButton) predictButton.disabled = true;

      let step = 0;
      let progress = 18;
      status.textContent = statuses[0];
      bar.style.width = `${progress}%`;

      const stepEls = loader.querySelectorAll(".loader-steps span");
      const timer = setInterval(() => {
        step = Math.min(step + 1, statuses.length - 1);
        progress = Math.min(progress + 23, 92);
        status.textContent = statuses[step];
        bar.style.width = `${progress}%`;
        stepEls.forEach((el, index) => el.classList.toggle("active", index <= step));
        if (step >= statuses.length - 1) clearInterval(timer);
      }, 550);

      // Native submit avoids firing this handler twice. The overlay remains visible
      // until the server returns the result page.
      setTimeout(() => form.submit(), 900);
    });
  }

  // If the page is restored from browser back-forward cache, never leave the loader visible.
  window.addEventListener("pageshow", () => {
    if (loader) {
      loader.classList.remove("show");
      loader.setAttribute("aria-hidden", "true");
    }
    document.body.classList.remove("loader-open");
    if (predictButton) predictButton.disabled = false;
  });
});
